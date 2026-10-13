import math
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class _LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1,
                 pad_idx=0, dropout=0.3, tie_weights=False):
        super().__init__()
        # net struct
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout) # NOTE: lstm blocks only apply dropouts if num_layers > 1. To avoid overfitting this applies it even if num_layers = 1
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        # memory efficiency flag. If emb_dim = hid_dim, share the two layers
        if tie_weights: 
            assert embedding_dim == hidden_dim, "weight tying needs emb_dim == hidden_dim"
            self.fc.weight = self.embedding.weight

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.drop(self.embedding(x)), hidden)
        return self.fc(self.drop(out)), hidden


class _LMDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences  # list[list[int]], each already has <s>...</s>

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# Guarantees that sentences in a batch are all of the same size
def _collate(batch, pad_id):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded


class LSTMLM:
    def __init__(self, embedding_dim=128, hidden_dim=256, num_layers=1,
                 unk_threshold=0, device=None, tie_weights=False, dropout=0.3):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.unk_threshold = unk_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.pad_token, self.bos, self.eos, self.unk_token = "<pad>", "<s>", "</s>", "<unk>"
        self.word2idx = {}
        self.idx2word = []
        self.tie_weights = tie_weights
        self.dropout = dropout
        self.model = None

    # ---------------- vocab + data prep ----------------

    def _build_vocab(self, tokenized_sentences):
        freq = Counter(w for sent in tokenized_sentences for w in sent)
        specials = [self.pad_token, self.bos, self.eos, self.unk_token] # special tok set
        words = [w for w, c in freq.items() if c > self.unk_threshold]
        self.idx2word = specials + sorted(words)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.pad_id = self.word2idx[self.pad_token]
        self.unk_id = self.word2idx[self.unk_token]

    def _to_ids(self, tokens):
        return [self.word2idx.get(w, self.unk_id) for w in tokens]

    # ---------------- training ----------------

    def train(self, train_sent, val_sent, epochs=20, batch_size=32, lr=1e-3,
              patience=3, model_path="lstm_lm_best.pt", verbose=True):
        self._build_vocab(train_sent)  # vocab from train only; val OOV words -> <unk>
        vocab_size = len(self.idx2word)

        t_sequences = [self._to_ids([self.bos] + sent + [self.eos]) for sent in train_sent]
        v_sequences = [self._to_ids([self.bos] + sent + [self.eos]) for sent in val_sent]

        train_ds = _LMDataset(t_sequences)
        val_ds = _LMDataset(v_sequences)  

        t_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: _collate(b, self.pad_id), # apply collate fun to both loaders
        )
        v_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: _collate(b, self.pad_id),
        )

        self.model = _LSTMNet(
            vocab_size, self.embedding_dim, self.hidden_dim,
            self.num_layers, pad_idx=self.pad_id, tie_weights=self.tie_weights, dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) # Adam optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id) # CE Loss (next tok classification task)
        # LR scheduler, val loss doesn't improve in 2 epochs, cut the lr in half
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        best_val_loss = float("inf")
        trigger_times = 0

        for epoch in range(1, epochs + 1):
            # --- train loop ---
            self.model.train()
            total_loss, n_batches = 0.0, 0
            for batch in t_loader:
                batch = batch.to(self.device)
                inputs, targets = batch[:, :-1], batch[:, 1:]

                optimizer.zero_grad()
                logits, _ = self.model(inputs)
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0) # clip gradient for stability (known problem for LSTM-type models)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            train_loss = total_loss / max(n_batches, 1)

            # --- validation ---
            self.model.eval()
            val_loss, n_val_batches = 0.0, 0
            with torch.no_grad():
                for batch in v_loader:
                    batch = batch.to(self.device)
                    inputs, targets = batch[:, :-1], batch[:, 1:]
                    logits, _ = self.model(inputs)
                    v_loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                    val_loss += v_loss.item()
                    n_val_batches += 1
            val_loss /= max(n_val_batches, 1)

            if verbose:
                print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} "
                      f"train_ppl={math.exp(train_loss):.2f}  "
                      f"val_loss={val_loss:.4f}  val_ppl={math.exp(val_loss):.2f}")

            scheduler.step(val_loss)
            # --- early stopping + checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self._checkpoint_dict(), model_path)  # full checkpoint, not just weights
                if verbose:
                    print(f"  --> best model saved (val_loss={best_val_loss:.4f})")
                trigger_times = 0
            else:
                trigger_times += 1
                if verbose:
                    print(f"  --> no improvement ({trigger_times}/{patience})")
                if trigger_times >= patience:
                    if verbose:
                        print("  --> early stopping triggered")
                    break

        # reload the best checkpoint
        self.load(model_path)

    def _checkpoint_dict(self):
        return {
            "state_dict": self.model.state_dict(),
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
        }

    def save(self, path):
        torch.save(self._checkpoint_dict(), path)

    def load(self, path):
        """
        Reload a checkpoint saved with .save() (produced by train()'s
        checkpointing).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.word2idx = ckpt["word2idx"]
        self.idx2word = ckpt["idx2word"]
        self.pad_id = ckpt["pad_id"]
        self.unk_id = ckpt["unk_id"]
        self.embedding_dim = ckpt["embedding_dim"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_layers = ckpt["num_layers"]

        vocab_size = len(self.idx2word)
        self.model = _LSTMNet(
            vocab_size, self.embedding_dim, self.hidden_dim,
            self.num_layers, pad_idx=self.pad_id
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self

    # --- scoring (complies decode) ---

    def logscore(self, word, context):
        """log2 P(word | context). context = list of preceding tokens (no <s>/</s>)."""
        seq = [self.bos] + list(context)
        ids = torch.tensor([self._to_ids(seq)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(ids)
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)  # distribution for the next token
        word_id = self.word2idx.get(word, self.unk_id)
        nat_logprob = log_probs[word_id].item()
        return nat_logprob / math.log(2)  # convert ln -> log2 to be consistent

    def score_sentence(self, tokens):

        seq = [self.bos] + list(tokens) + [self.eos] # <s> toks </s>
        ids = self._to_ids(seq)
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        target_ids = ids[1:]

        with torch.no_grad():
            logits, _ = self.model(input_ids)
        log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

        total_nats = sum(log_probs[i, target_ids[i]].item() for i in range(len(target_ids)))
        total_log2 = total_nats / math.log(2)
        n_words = len(tokens) + 1  # +1 for </s>
        return total_log2, total_log2 / n_words
