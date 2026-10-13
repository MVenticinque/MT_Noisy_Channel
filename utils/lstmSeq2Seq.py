import math
import random
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class _Seq2SeqNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512, num_layers=2,
                 pad_idx=0, dropout=0.3, tie_weights=True):
        super().__init__()
        self.pad_idx, self.num_layers, self.hidden_dim = pad_idx, num_layers, hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.bridge_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.bridge_c = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)   # Luong
        self.combine = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        if tie_weights:
            assert embedding_dim == hidden_dim, "weight tying needs emb_dim == hidden_dim"
            self.fc.weight = self.embedding.weight

    def encode(self, src):
        out, (hn, cn) = self.encoder(self.drop(self.embedding(src)))   # out: (B,S,2H)
        hn = hn.view(self.num_layers, 2, -1, self.hidden_dim)          # (L,2,B,H)
        cn = cn.view(self.num_layers, 2, -1, self.hidden_dim)
        hn = torch.cat([hn[:, 0], hn[:, 1]], dim=-1)                   # (L,B,2H)
        cn = torch.cat([cn[:, 0], cn[:, 1]], dim=-1)
        state = (torch.tanh(self.bridge_h(hn)).contiguous(),
                 torch.tanh(self.bridge_c(cn)).contiguous())
        return out, state

    def _attend(self, dec_out, enc_out, src_mask):
        scores = self.attn(dec_out) @ enc_out.transpose(1, 2)         # (B,T,S)
        scores = scores.masked_fill(~src_mask.unsqueeze(1), -1e9)
        ctx = scores.softmax(-1) @ enc_out                             # (B,T,2H)
        return torch.cat([dec_out, ctx], dim=-1)                       # (B,T,3H)

    def forward(self, src, trg_in):
        """Teacher-forced training pass. trg_in already starts with <s>."""
        enc_out, state = self.encode(src)
        dec_out, _ = self.decoder(self.drop(self.embedding(trg_in)), state)
        combined = self._attend(dec_out, enc_out, src != self.pad_idx)
        return self.fc(self.drop(torch.tanh(self.combine(combined))))

    def step(self, y_prev, state, enc_out, src_mask):
        """One decoding step. y_prev: (B,) -> logits (B,V), new state."""
        emb = self.drop(self.embedding(y_prev)).unsqueeze(1)           # (B,1,E)
        dec_out, state = self.decoder(emb, state)                       # (B,1,H)
        combined = self._attend(dec_out, enc_out, src_mask)
        return self.fc(self.drop(torch.tanh(self.combine(combined)))).squeeze(1), state


def _collate_pairs(batch, pad_id):
    srcs, tgts = zip(*batch)
    S, T = max(len(s) for s in srcs), max(len(t) for t in tgts)
    src = torch.full((len(batch), S), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), T), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, :len(s)], tgt[i, :len(t)] = torch.tensor(s), torch.tensor(t)
    return src, tgt


class Seq2SeqMT:
    def __init__(self, embedding_dim=512, hidden_dim=512, num_layers=2,
                 unk_threshold=1, device=None, tie_weights=True, dropout=0.3):
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.num_layers, self.dropout = num_layers, dropout
        self.unk_threshold = unk_threshold
        self.tie_weights = tie_weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_token, self.bos, self.eos, self.unk_token = "<pad>", "<s>", "</s>", "<unk>"
        self.word2idx, self.idx2word = {}, []
        self.model = None

    # ---------------- vocab (joint over src+tgt) ----------------
    def _build_vocab(self, train_pairs):
        freq = Counter(w for s, t in train_pairs for w in list(s) + list(t))
        specials = [self.pad_token, self.bos, self.eos, self.unk_token]
        words = [w for w, c in freq.items() if c > self.unk_threshold]
        self.idx2word = specials + sorted(words)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.pad_id = self.word2idx[self.pad_token]
        self.unk_id = self.word2idx[self.unk_token]
        self.bos_id, self.eos_id = self.word2idx[self.bos], self.word2idx[self.eos]

    def _to_ids(self, tokens):
        return [self.word2idx.get(w, self.unk_id) for w in tokens]

    # ---------------- training (mirrors LSTMLM.train) ----------------
    def train(self, train_pairs, val_pairs, epochs=15, batch_size=64, lr=1e-3,
              patience=3, model_path="seq2seq_best.pt", verbose=True):
        """train_pairs / val_pairs: list[(src_tokens, tgt_tokens)] -- words or BPE pieces."""
        self._build_vocab(train_pairs)
        vocab_size = len(self.idx2word)

        prep = lambda pairs: [
            (self._to_ids(s) + [self.eos_id],                 # src: tokens + </s>
             [self.bos_id] + self._to_ids(t) + [self.eos_id]) # tgt: <s> + tokens + </s>
            for s, t in pairs]
        t_data, v_data = prep(train_pairs), prep(val_pairs)

        rng = random.Random(1)
        def batches(data, bs, shuffle):
            # length-bucketing (big speed win for LSTMs) + shuffled batch order
            order = list(range(len(data)))
            key = (lambda i: len(data[i][0]) + len(data[i][1]) +
                   (rng.random() if shuffle else 0.0))
            order.sort(key=key)
            chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
            if shuffle:
                rng.shuffle(chunks)
            for c in chunks:
                yield _collate_pairs([data[i] for i in c], self.pad_id)

        self.model = _Seq2SeqNet(
            vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers,
            pad_idx=self.pad_id, tie_weights=self.tie_weights, dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1)
        best_val_loss, trigger_times = float("inf"), 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, n = 0.0, 0
            for src, tgt in batches(t_data, batch_size, shuffle=True):
                src, tgt = src.to(self.device), tgt.to(self.device)
                optimizer.zero_grad()
                logits = self.model(src, tgt[:, :-1])
                loss = criterion(logits.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item(); n += 1
            train_loss = total_loss / max(n, 1)

            self.model.eval()
            val_loss, n = 0.0, 0
            with torch.no_grad():
                for src, tgt in batches(v_data, batch_size, shuffle=False):
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    logits = self.model(src, tgt[:, :-1])
                    v = criterion(logits.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
                    val_loss += v.item(); n += 1
            val_loss /= max(n, 1)

            if verbose:
                print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_ppl={math.exp(val_loss):.2f}")
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self._checkpoint_dict(), model_path)
                trigger_times = 0
                if verbose:
                    print(f"  --> best model saved (val_loss={best_val_loss:.4f})")
            else:
                trigger_times += 1
                if verbose:
                    print(f"  --> no improvement ({trigger_times}/{patience})")
                if trigger_times >= patience:
                    if verbose:
                        print("  --> early stopping triggered")
                    break

        self.load(model_path)   # reload the BEST checkpoint, like LSTMLM

    # ---------------- checkpointing (same as LSTMLM) ----------------
    def _checkpoint_dict(self):
        return {"state_dict": self.model.state_dict(), "word2idx": self.word2idx,
                "idx2word": self.idx2word, "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
                "dropout": self.dropout, "tie_weights": self.tie_weights,
                "pad_id": self.pad_id, "unk_id": self.unk_id,
                "bos_id": self.bos_id, "eos_id": self.eos_id}

    def save(self, path):
        torch.save(self._checkpoint_dict(), path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for k in ("word2idx", "idx2word", "embedding_dim", "hidden_dim",
                  "num_layers", "dropout", "tie_weights",
                  "pad_id", "unk_id", "bos_id", "eos_id"):
            setattr(self, k, ckpt[k])
        self.model = _Seq2SeqNet(
            len(self.idx2word), self.embedding_dim, self.hidden_dim, self.num_layers,
            pad_idx=self.pad_id, tie_weights=self.tie_weights, dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self

    # ---------------- translation ----------------
    @torch.no_grad()
    def translate(self, token_lists, beam_size=5, max_len=60, alpha=0.7, batch_size=32):
        """token_lists: list of source token lists (words or BPE pieces).
        Returns list of token lists. beam_size<=1 -> batched greedy (fast sanity checks)."""
        self.model.eval()
        if beam_size <= 1:
            return self._greedy(token_lists, max_len, batch_size)
        return [self._beam(t, beam_size, max_len, alpha) for t in token_lists]

    def _greedy(self, token_lists, max_len, batch_size):
        out = []
        for i in range(0, len(token_lists), batch_size):
            chunk = token_lists[i:i + batch_size]
            src = _collate_pairs([(self._to_ids(t) + [self.eos_id], [self.bos_id])
                                  for t in chunk], self.pad_id)[0].to(self.device)
            enc_out, state = self.model.encode(src)
            mask = src != self.model.pad_idx
            ys = torch.full((len(chunk),), self.bos_id, dtype=torch.long, device=self.device)
            hyps, done = [[] for _ in chunk], [False] * len(chunk)
            for _ in range(max_len):
                logits, state = self.model.step(ys, state, enc_out, mask)
                ys = logits.argmax(-1)
                for b, y in enumerate(ys.tolist()):
                    if not done[b]:
                        if y == self.eos_id:
                            done[b] = True
                        else:
                            hyps[b].append(self.idx2word[y])
                if all(done):
                    break
            out += hyps
        return out

    def _beam(self, tokens, k, max_len, alpha):
        src = torch.tensor([self._to_ids(tokens) + [self.eos_id]],
                           dtype=torch.long, device=self.device)
        enc_out, state = self.model.encode(src)
        mask = src != self.model.pad_idx
        beam, finished = [(0.0, [self.bos_id], state)], []
        for _ in range(max_len):
            if not beam:
                break
            cands = []
            for lp, toks, st in beam:
                logits, st2 = self.model.step(
                    torch.tensor([toks[-1]], device=self.device), st, enc_out, mask)
                logp = torch.log_softmax(logits[0], dim=-1)
                for v, i in zip(*logp.topk(k)):
                    cands.append((lp + v.item(), toks + [i.item()], st2, i.item() == self.eos_id))
            cands.sort(key=lambda c: -c[0])
            beam = []
            for lp, toks, st, done in cands:
                if done:
                    finished.append((lp, toks))
                elif len(beam) < k:
                    beam.append((lp, toks, st))
        pool = finished or [(lp, toks) for lp, toks, _ in beam]
        # alpha = the seq2seq analog of your word_bonus: fixes length bias
        best = max(pool, key=lambda p: p[0] / max(len(p[1]) - 1, 1) ** alpha)
        ids = best[1][1:]
        if ids and ids[-1] == self.eos_id:
            ids = ids[:-1]
        return [self.idx2word[i] for i in ids]


def eval_seq2seq(mt, val_df, sp=None, n_test=200, seed=0,
                 beam_size=5, alpha=0.7, verbose=True):
    """sp=None -> word-level model; sp=sentencepiece model -> BPE model.
    Same sampling/refs/scoring as run_comparison() and the opus-mt eval."""
    df = val_df.dropna(subset=["sent_it", "sent_en"]).sample(
        n=min(n_test, len(val_df)), random_state=seed)
    refs = [" ".join(tok_en(s)) for s in df["sent_en"]]

    if sp is None:
        srcs = [tok_it(s) for s in df["sent_it"]]
        raw = mt.translate(srcs, beam_size=beam_size, alpha=alpha)
        hyps = [" ".join(h) for h in raw]
    else:
        srcs = [sp.encode(" ".join(tok_it(s)), out_type=str) for s in df["sent_it"]]
        pieces = mt.translate(srcs, beam_size=beam_size, alpha=alpha)
        hyps = [" ".join(tok_en(sp.decode(h))) for h in pieces]  # same detok->tok_en as opus row

    t0 = time.time()
    _ = 0  # (timing done inside translate if you want it; kept simple)
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none", force=True).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    lr = sum(len(h.split()) for h in hyps) / sum(len(r.split()) for r in refs)
    if verbose:
        print(f"seq2seq: BLEU {bleu:.2f}  chrF {chrf:.2f}  len-ratio {lr:.2f}")
    return dict(bleu=bleu, chrf=chrf, length_ratio=lr, hyps=hyps)