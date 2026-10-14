import math

import torch
from sacremoses import MosesDetokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaWordLM:
    def __init__(self, name="google/gemma-3-1b-pt", device=None, dtype=torch.bfloat16,
                 batch_size=16, eos_mode="zero", max_cache=2_000_000):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.eos = "</s>" # eos string used by decode()
        self.eos_mode = eos_mode
        self.batch_size = batch_size # seq per forward pass
        self.max_cache = max_cache
        self.detok = MosesDetokenizer("en").detokenize # convert list of tokens back to strings
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0 # padding
        self._ln2 = math.log(2)
        self._ids_cache = {} # tuple(tokens) -> token ids (BOS + text)
        self._tot_cache = {} # tuple(ids)    -> ln P(ids[1:] | ids[0])

    def clear_cache(self):
        self._ids_cache.clear()
        self._tot_cache.clear()

    # --- tokenization ---
    def _text(self, tokens):
        s = self.detok(list(tokens))
        return s[:1].upper() + s[1:]  # Gemma is cased, our tokens are lowercase

    def _ids_many(self, token_lists):
        keys = [tuple(t) for t in token_lists]
        missing = list(dict.fromkeys(k for k in keys if k not in self._ids_cache))
        if missing:
            enc = self.tok([self._text(k) for k in missing]).input_ids # one batched call for all missing tok
            for k, ids in zip(missing, enc):
                self._ids_cache[k] = list(ids)
        return [self._ids_cache[k] for k in keys]

    def _ids(self, tokens):
        return self._ids_many([tokens])[0]

    def _end_id(self):
        if self.eos_mode == "eos":
            return self.tok.eos_token_id
        return self.tok("\n", add_special_tokens=False).input_ids[-1]

    # --- model call ---
    @torch.no_grad()
    def _forward_totals(self, chunk):

        L = max(len(s) for s in chunk)
        if L < 2: # if longest seq in batch is smaller than 2, there's nothing after bos
            return [0.0] * len(chunk)
        x = torch.tensor([s + [self.pad_id] * (L - len(s)) for s in chunk], device=self.device) # right pad seq
        attn = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in chunk], device=self.device) # att on real tok 1, padded tok 0
        logits = self.model(input_ids=x, attention_mask=attn).logits[:, :-1].float()
        tl = logits.gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        return (tl * attn[:, 1:].float()).sum(1).tolist()

    def _totals(self, id_lists):
        if len(self._tot_cache) > self.max_cache:
            self.clear_cache()
        keys = [tuple(s) for s in id_lists]
        missing = sorted(set(k for k in keys if k not in self._tot_cache), key=len)
        for b in range(0, len(missing), self.batch_size):
            chunk = missing[b:b + self.batch_size]
            for k, v in zip(chunk, self._forward_totals([list(c) for c in chunk])):
                self._tot_cache[k] = v
        return [self._tot_cache[k] for k in keys]

    def logscore_many(self, words, context):
        words = list(words)
        if not words:
            return []
        context = list(context)
        real = [w for w in dict.fromkeys(words) if w != self.eos]
        ids_real = self._ids_many([context + [w] for w in real])
        base_ids = self._ids(context)

        want_eos = self.eos in words
        eos_ids = [base_ids + [self._end_id()]] if (want_eos and self.eos_mode != "zero") else []
        tot = self._totals(ids_real + eos_ids + [base_ids])
        base = tot[-1]

        res = {w: (t - base) / self._ln2 for w, t in zip(real, tot)}
        if want_eos:
            res[self.eos] = (tot[len(real)] - base) / self._ln2 if eos_ids else 0.0
        return [res[w] for w in words]

    def logscore(self, word, context):
        return self.logscore_many([word], context)[0]

    def score_sentence(self, tokens, add_eos=True):
        """
        Returns (total_log2, per_word_log2) for a whole sentence.
        per_word_log2 has one entry per word, plus one for </s> when add_eos=True,
        so len(per_word_log2) == len(tokens) + 1 (the convention perplexity() uses).
        With eos_mode="zero" the </s> entry is 0.0.

        tokens: list of lowercase tokens, or a space-separated string.
        """
        if isinstance(tokens, str):
            tokens = tokens.split()
        tokens = list(tokens)
        per_word = []
        if tokens:
            tot = self._totals(self._ids_many([tokens[:k] for k in range(len(tokens) + 1)]))
            per_word = [(tot[k + 1] - tot[k]) / self._ln2 for k in range(len(tokens))]
        if add_eos:
            per_word.append(self.logscore(self.eos, tokens))
        return sum(per_word), per_word