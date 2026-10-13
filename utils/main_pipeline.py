import math
import pickle
import re
from collections import defaultdict
import nltk
import eflomal

from huggingface_hub import hf_hub_download
import sentencepiece as spm

import sacrebleu

MIN_LEN, MAX_LEN, MAX_RATIO = 4, 20, 2.5

SRC_FILE, TGT_FILE = "train.it", "train.en"
FWD_FILE, REV_FILE = "forward.align", "reverse.align"
TABLES_FILE = "tables.pkl"


# --------------------------------------------------------------------------
# 1. Tokenization
# --------------------------------------------------------------------------

_IT_TOKEN = re.compile(r"\w+'(?=\w)|\w+(?:[-/.]\w+)*|[^\w\s]")

# Three diff kinds of tokenization rules for Italian (spaces are stripped):
# 1) split along apostrophes, 2) keep compound words together, 3) isolate punctuation as single tokens

# 1: \w+'(?=\w)
# \w+ matches one or more alphanum chars, ' followed by a single apostrophe,
# (?=\w) looks after the ' and checks if the next char is alphanum. If it is, cut

# 2: \w+(?:[-/.]\w+)*
# \w+ matches one or more alphanum chars, (?:...)* repeats 0 or more times:
# a single '-', '/', or '.' followed by one or more alphanum chars (\w+)

# 3: [^\w\s]
# [^...] NOT matches any single char that is neither alphanum (\w) nor whitespace (\s) (basically punctuation)



def tok_it(s):
    s = s.lower().replace("\u2019", "'").replace("\u2018", "'") # we work in lowercase, and replace ` with '
    return _IT_TOKEN.findall(s) # apply tok rule


def tok_en(s):
    try:
        return nltk.word_tokenize(s.lower(), language="english")
    except LookupError: # if punkt_tab not already installed
        nltk.download("punkt")
        nltk.download("punkt_tab")
        return nltk.word_tokenize(s.lower(), language="english")

# Pretrained Sub Word tokenizer
sp_it = spm.SentencePieceProcessor(model_file=hf_hub_download("Helsinki-NLP/opus-mt-it-en", "source.spm")) 
sp_en = spm.SentencePieceProcessor(model_file=hf_hub_download("Helsinki-NLP/opus-mt-it-en", "target.spm"))

# sp tok for more granularity
def it_pieces(raw):  return sp_it.encode(" ".join(tok_it(raw)), out_type=str)
def en_pieces(raw):  return sp_en.encode(" ".join(tok_en(raw)), out_type=str)
def pieces_to_words(pieces): return tok_en(sp_en.decode(pieces))   # pieces -> BLEU-ready words


# --------------------------------------------------------------------------
# 2. Data preparation
# --------------------------------------------------------------------------
def prepare_data(csv_path, src_file=SRC_FILE, tgt_file=TGT_FILE,
                 n_sample=200_000, seed=0):
    import pandas as pd

    # load df
    df = pd.read_csv(csv_path).dropna(subset=["sent_en", "sent_it"])
    # filter using max len and min len to dont have too long sequences + keep max_ratio between ita and en translations small
    en_len = df["sent_en"].str.split().str.len()
    it_len = df["sent_it"].str.split().str.len()
    keep = (
        en_len.between(MIN_LEN, MAX_LEN)
        & it_len.between(MIN_LEN, MAX_LEN)
        & (en_len / it_len <= MAX_RATIO)
        & (it_len / en_len <= MAX_RATIO)
    )
    df = df[keep]

    # downsample and reproducibility (filtered df is ~ 600k)
    if n_sample and len(df) > n_sample:
        df = df.sample(n_sample, random_state=seed)

    # word-level tok -> write parallel files
    n = 0
    with open(src_file, "w", encoding="utf-8") as f_it, \
         open(tgt_file, "w", encoding="utf-8") as f_en:
        for it_sent, en_sent in zip(df["sent_it"], df["sent_en"]):
            it_t, en_t = tok_it(it_sent), tok_en(en_sent)
            if not it_t or not en_t: 
                continue
            f_it.write(" ".join(it_t) + "\n")
            f_en.write(" ".join(en_t) + "\n")
            n += 1
    print(f"wrote {n} sentence pairs to {src_file} / {tgt_file}")

    # subword tok -> read the word-tokenized files and encode into .sp files
    nn = 0
    with open(src_file, encoding="utf-8") as fi, open(tgt_file, encoding="utf-8") as fe, \
        open(src_file+'.sp', "w", encoding="utf-8") as spi, open(tgt_file+'.sp', "w", encoding="utf-8") as spe:
        for a, b in zip(fi, fe):
            spi.write(" ".join(sp_it.encode(a.strip(), out_type=str)) + "\n")
            spe.write(" ".join(sp_en.encode(b.strip(), out_type=str)) + "\n")
            nn += 1
    print(f"wrote {nn} sentence pairs to {src_file+'.sp'} / {tgt_file+'.sp'}")


# --------------------------------------------------------------------------
# 3. Alignment in both directions using eflomal
# --------------------------------------------------------------------------
def run_eflomal(src_file=SRC_FILE, tgt_file=TGT_FILE,
                fwd_file=FWD_FILE, rev_file=REV_FILE):
    # initialize the aligner to create align files
    aligner = eflomal.Aligner() # model used 3, hence IBM1 init with HMM and fertility
    with open(src_file, encoding="utf-8") as src_data, \
         open(tgt_file, encoding="utf-8") as trg_data:
        aligner.align(
            src_data, trg_data,
            links_filename_fwd=fwd_file, # from it -> en (it_idx-en_idx)
            links_filename_rev=rev_file, # from en -> en (written as it_idx-en_idx)
        )


# --------------------------------------------------------------------------
# 4. Translation tables from sym links
# --------------------------------------------------------------------------
def build_tables(src_file=SRC_FILE, tgt_file=TGT_FILE,
                 fwd_file=FWD_FILE, rev_file=REV_FILE,
                 mode="intersect", min_prob=1e-3, max_cands=20):
    """
    Returns (p_e_given_f, p_f_given_e).

    p_e_given_f[f][e]  : candidate generator. The key None means "f is deleted"
                         (f had no link), which handles articles/clitics.
    p_f_given_e[f][e]  : count(f, e) / (total corpus occurrences of e).
                         Using total occurrences, not just aligned ones, keeps
                         frequent words like 'the' honest (they are often unaligned).

    mode: "intersect" (high precision, default), "union", or "fwd".
    """
    c = defaultdict(lambda: defaultdict(int)) # c[italian][english or None]
    f_tot = defaultdict(int) # all occurrences of word f_i in italian in corpus
    e_tot = defaultdict(int) # all occurrences of word e_i in english in corpus

    with open(src_file, encoding="utf-8") as fs, open(tgt_file, encoding="utf-8") as ft, \
         open(fwd_file, encoding="utf-8") as ff, open(rev_file, encoding="utf-8") as fr:
        for s, t, a, b in zip(fs, ft, ff, fr):  
            # s = source sentence
            # t = target sentence
            # a = forward align
            # b = backward align
            src, tgt = s.split(), t.split() # split senteces into toks
            A, B = set(a.split()), set(b.split()) # split align strings into sets of it_idx, en_idx (for both f and b)
            if mode == "intersect": # Keep only links present in both A and B
                links = A & B
            elif mode == "union": # Keep the union
                links = A | B
            elif mode == "fwd": # Use only fwd
                links = A
            else:
                raise ValueError(f"unknown mode {mode!r}")

            for w in tgt: 
                e_tot[w] += 1 # count instances of target word w

            aligned_src = set()
            for pair in links:
                i, j = map(int, pair.split("-")) # the links are always src-tgt
                c[src[i]][tgt[j]] += 1 # Look up at c[src_word][tgt_word] and increase counter by 1
                f_tot[src[i]] += 1 # count instances of src w
                aligned_src.add(i) # keep track of aligned idxs
            for i, f in enumerate(src): # unaligned toks will be del during transl. hence w_idx -> None
                if i not in aligned_src:
                    c[f][None] += 1
                    f_tot[f] += 1

    p_e_given_f, p_f_given_e = {}, {}
    for f, d in c.items(): # iterate through all ita words and aligned references in english

        # prune them using conditonal probability
        ranked = sorted(((e, n / f_tot[f]) for e, n in d.items()),
                        key=lambda kv: -kv[1]) 
        ranked = [(e, p) for e, p in ranked if p >= min_prob][:max_cands]
        if not ranked:
            continue

        p_e_given_f[f] = dict(ranked)
        p_f_given_e[f] = {e: d[e] / e_tot[e] for e, _ in ranked if e is not None}
    return p_e_given_f, p_f_given_e # return the pruned tabs

# Helper functions
def save_tables(p_ef, p_fe, path=TABLES_FILE):
    with open(path, "wb") as f:
        pickle.dump((p_ef, p_fe), f)


def load_tables(path=TABLES_FILE):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# 5. Decoder
# --------------------------------------------------------------------------
class Hypothesis:
    __slots__ = ("covered", "tokens", "score", "last_src_idx")

    def __init__(self, covered, tokens, score, last_src_idx=-1):
        self.covered = covered            # frozenset of covered source tok indices
        self.tokens = tokens              # english toks output so far
        self.score = score                # cumulative log2 score
        self.last_src_idx = last_src_idx  # for the distortion term (reordering)


COPY_PROB = 1e-3   # probability used for copy-through of unknown source words


def get_candidates(f, p_e_given_f, top_k=5):
    """Top-k (e, P(e|f)) for source word f. e may be None (= delete f).
    Unknown words are copied through unchanged."""
    dist = p_e_given_f.get(f)
    if not dist:
        return [(f, COPY_PROB)]
    return sorted(dist.items(), key=lambda kv: -kv[1])[:top_k]


def decode(
    source_tokens, # toks in italian
    # Align tables
    p_e_given_f, 
    p_f_given_e,

    lm, # languange model. It has to expose logscore(word, ctx)
    beam_size=10, # max num of hypothesis kept at each step
    candidates_per_word=5, # num of topk words evaluated at each step for each hyp
    lm_order=None, # legnth of context. None: full ctx.
    w_lm=1.0, # lambda wegiht for lm
    w_ef=1.0, # lambda weight for fwd tab
    w_fe=0.5, # lambda weight for bak tab
    word_bonus=0.0, # reward per emitted word
    distortion_penalty=1.0,
    threshold=None, # threshold for pruning. None: no prune
    warmup_steps=3, # warmup steps before pruning
):
    """
    Each step picks one uncovered source position j and either emits one English
    word for it or deletes it. After n_src steps every source word is covered, so
    all hypotheses finish together and are compared after adding the </s> score.

    distortion_penalty: alpha in -alpha * |j - last_j - 1| (log2 units). Around 1
        similar word order. 0 = free reordering.
    Returns (best_tokens, best_score, top5_list_of_Hypothesis).
    """
    n_src = len(source_tokens)
    if n_src == 0: # if provided sentence is empty, return empty output
        return [], 0.0, []

    def lm_context(tokens):
        return tokens if lm_order is None else tokens[-lm_order:]

    lm_cache = {} # dict shared across all steps. Compute the pair (ctx, next_tok) only once per call

    def lm_score(word, tokens):
        ctx = tuple(lm_context(tokens))
        key = (ctx, word)
        v = lm_cache.get(key)
        # this runs if not in lm_cache
        if v is None:
            v = lm.logscore(word, list(ctx))
            lm_cache[key] = v
        return v

    beam = [Hypothesis(frozenset(), [], 0.0, -1)] # emtpy frozenset (nothing translated at the start of decoding), hence empty output, 0 cum_score, -1 distortion

    for step in range(1, n_src + 1): # one step per word
        expanded = {}
        for hyp in beam: # iterate through each hyp still in beam
            for j in range(n_src):
                if j in hyp.covered: # skip already used toks
                    continue
                f = source_tokens[j]
                dist_score = -distortion_penalty * abs(j - hyp.last_src_idx - 1)
                for e, p_ef in get_candidates(f, p_e_given_f, candidates_per_word):
                    if e is None:                                   # if e is None delete f, it tranlsates to nothing
                        score = hyp.score + w_ef * math.log2(p_ef) + dist_score # no lm term since nothing is emitted
                        toks = hyp.tokens
                    else:
                        if e == f and f not in p_e_given_f:         # unk tok
                            p_fe = COPY_PROB
                        else:
                            p_fe = p_f_given_e.get(f, {}).get(e, 1e-6)
                        score = (hyp.score # old score  plus
                                 + w_lm * lm_score(e, hyp.tokens) # how likely the lm thinks e comes after toks
                                 + w_ef * math.log2(p_ef) # P(e|f)
                                 + w_fe * math.log2(p_fe) # P(f|e)
                                 + word_bonus # Flat bonus to balance the lm outputting small sentences
                                 + dist_score) # dist penalty
                        toks = hyp.tokens + [e] # New list per hyp
                    covered = hyp.covered | {j} # new frozenset as union of prev + j
                    # If two hyps have produced the same output, keep only the best
                    key = (covered, tuple(toks))
                    old = expanded.get(key)
                    if old is None or score > old.score:
                        expanded[key] = Hypothesis(covered, toks, score, j)
        # Pruning
        cands = sorted(expanded.values(), key=lambda h: -h.score)
        if threshold is not None and step > warmup_steps and cands:
            best = cands[0].score
            cands = [h for h in cands if h.score >= best - threshold]
        beam = cands[:beam_size] # keep only beam_size candidates

    finished = []
    for h in beam:
        eos = w_lm * lm.logscore(lm.eos, list(lm_context(h.tokens))) # add eos to the mix
        finished.append(Hypothesis(h.covered, h.tokens, h.score + eos, h.last_src_idx)) # new hyp created
    finished.sort(key=lambda h: -h.score) 
    return finished[0].tokens, finished[0].score, finished[:5]


# --------------------------------------------------------------------------
# 6. Tuning (coordinate ascent on dev BLEU)
# --------------------------------------------------------------------------
def tune(dev_pairs, p_ef, p_fe, lm, rounds=2, verbose=True, init_params=None, grid=None, **decode_kw):
    """
    dev_pairs: list of (italian_raw, english_raw) sentences.
    Returns the best parameter dict for decode(..., **params).
    """

    src = [tok_it(it) for it, _ in dev_pairs]
    refs = [" ".join(tok_en(en)) for _, en in dev_pairs]

    # initial param point and grid search entries
    params = init_params or dict(w_lm=0.5, w_fe=0.0, word_bonus=2.0, distortion_penalty=0.0)
    grid = grid or dict(
        w_lm=[0.75, 1.0, 1.25],
        w_fe=[0.25, 0.5, 1.0],
        word_bonus=[4.0, 8.0, 10.0],
        distortion_penalty=[0.5, 1.0, 2.0],
    )
    cache = {} # keep track of bleu for a given set of params

    def bleu(p):
        key = tuple(sorted(p.items()))
        if key not in cache: # if set of params not yet explored, compute bleu and append
            hyps = [" ".join(decode(s, p_ef, p_fe, lm, **p, **decode_kw)[0]) for s in src]
            cache[key] = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score
        return cache[key]

    best = bleu(params)
    if verbose:
        print(f"start BLEU {best:.2f} {params}") # initial bleu
    for r in range(rounds):
        improved = False
        for name, values in grid.items(): # iterate through grid
            for v in values:
                trial = dict(params, **{name: v})
                score = bleu(trial)
                if score > best + 1e-9: # 1e-9 if score is a real improvement over best
                    best, params, improved = score, trial, True
                    if verbose:
                        print(f"round {r + 1}: BLEU {best:.2f} {params}")
        if not improved: # if full grid search set leads to no improvement, break
            break
    return params
