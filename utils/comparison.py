import time
import sacrebleu


def perplexity(lm, tokenized_sentences):
    total_log2, n_words = 0.0, 0
    for toks in tokenized_sentences:
        t, _ = lm.score_sentence(toks)
        total_log2 += t
        n_words += len(toks) + 1  # +1 for </s>, matches score_sentence's convention
    return 2 ** (-total_log2 / n_words)


def run_comparison(val_df, p_ef, p_fe, lms, params=None, n_test=200,
                   decode_kw=None, show_examples=3, seed=0):
    """
    val_df: dataframe with 'sent_it' / 'sent_en' columns
    lms: dict[name -> lm_object_or_config] -- see module docstring for the
         per-model override format.
    params / decode_kw: SHARED defaults, used for any entry in `lms` that
         doesn't supply its own "params"/"decode_kw" override.
    """
    default_params = params or dict(w_lm=0.6, w_fe=0.0, word_bonus=20.0, distortion_penalty=0.5)
    default_decode_kw = decode_kw or dict(beam_size=10, candidates_per_word=5)

    df = val_df.dropna(subset=["sent_it", "sent_en"]).sample(
        n=min(n_test, len(val_df)), random_state=seed
    )
    src_raw = df["sent_it"].tolist()
    ref_raw = df["sent_en"].tolist()

    src_tok = [tok_it(s) for s in src_raw]
    ref_tok = [tok_en(s) for s in ref_raw]
    refs_joined = [" ".join(r) for r in ref_tok]

    results = {}
    for name, entry in lms.items():
        if isinstance(entry, dict) and "lm" in entry:
            lm = entry["lm"]
            this_params = entry.get("params", default_params)
            this_decode_kw = entry.get("decode_kw", default_decode_kw)
        else:
            lm = entry
            this_params = default_params
            this_decode_kw = default_decode_kw

        print(f"\n=== {name} ===")
        print(f"params={this_params}  decode_kw={this_decode_kw}")

        ppl = perplexity(lm, ref_tok)  # LM's own fluency on the references, context for BLEU
        print(f"perplexity on references: {ppl:.2f}")

        t0 = time.time()
        hyps = []
        for toks in src_tok:
            best_tokens, _, _ = decode(toks, p_ef, p_fe, lm, **this_params, **this_decode_kw)
            hyps.append(" ".join(best_tokens))
        elapsed = time.time() - t0

        bleu = sacrebleu.corpus_bleu(hyps, [refs_joined], tokenize="none").score
        chrf = sacrebleu.corpus_chrf(hyps, [refs_joined]).score
        hyp_len = sum(len(h.split()) for h in hyps)
        ref_len = sum(len(r) for r in ref_tok)  # already tokenized, no .split() needed

        print(f"BLEU: {bleu:.2f}   chrF: {chrf:.2f}   length ratio (hyp/ref): {hyp_len/ref_len:.2f}")
        print(f"decode time: {elapsed:.1f}s total, {elapsed/len(src_tok)*1000:.0f}ms/sentence")

        results[name] = dict(bleu=bleu, chrf=chrf, ppl=ppl, time_s=elapsed,
                             length_ratio=hyp_len / ref_len, hyps=hyps,
                             params=this_params, decode_kw=this_decode_kw)

    print(f"\n=== qualitative examples (first {show_examples}) ===")
    for i in range(min(show_examples, len(src_raw))):
        print(f"\nSource: {' '.join(src_tok[i])}")
        print(f"Target: {' '.join(ref_tok[i])}")
        for name in lms:
            print(f"{name:>10}: {results[name]['hyps'][i]}")

    return results

def run_comparison_pieces(val_df, p_ef, p_fe, lms, params=None, n_test=200,
                    decode_kw=None, show_examples=3, seed=0):

    default_params = params or dict(w_lm=0.6, w_fe=0.0, word_bonus=20.0, distortion_penalty=0.5)
    default_decode_kw = decode_kw or dict(beam_size=10, candidates_per_word=8)

    df = val_df.dropna(subset=["sent_it", "sent_en"]).sample(
        n=min(n_test, len(val_df)), random_state=seed)
    df['tok_len'] = df.sent_it.apply(it_pieces).str.len()
    df = df[df['tok_len'] < 150]
    src_raw = df["sent_it"].tolist()
    ref_raw = df["sent_en"].tolist()

    src_tok = [it_pieces(s) for s in src_raw]
    refs_joined = [" ".join(tok_en(s)) for s in ref_raw] 
    ref_pieces = [en_pieces(s) for s in ref_raw]       

    results = {}
    for name, entry in lms.items():
        if isinstance(entry, dict) and "lm" in entry:
            lm = entry["lm"]
            this_params = entry.get("params", default_params)
            this_decode_kw = entry.get("decode_kw", default_decode_kw)
        else:
            lm = entry
            this_params = default_params
            this_decode_kw = default_decode_kw

        print(f"\n=== {name} ===")
        print(f"params={this_params}  decode_kw={this_decode_kw}")

        # not comparable to word-level rows
        ppl = perplexity(lm, ref_pieces)
        print(f"perplexity on references (per piece): {ppl:.2f}")

        t0 = time.time()
        hyps = []
        for toks in src_tok:
            best_tokens, _, _ = decode(toks, p_ef, p_fe, lm, **this_params, **this_decode_kw)
            hyps.append(" ".join(pieces_to_words(best_tokens)))   # pieces -> WORDS
        elapsed = time.time() - t0

        bleu = sacrebleu.corpus_bleu(hyps, [refs_joined], tokenize="none", force=True).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs_joined]).score
        hyp_len = sum(len(h.split()) for h in hyps)
        ref_len = sum(len(r.split()) for r in refs_joined)

        print(f"BLEU: {bleu:.2f}   chrF: {chrf:.2f}   length ratio (hyp/ref): {hyp_len/ref_len:.2f}")
        print(f"decode time: {elapsed:.1f}s total, {elapsed/len(src_tok)*1000:.0f}ms/sentence")

        results[name] = dict(bleu=bleu, chrf=chrf, ppl=ppl, time_s=elapsed,
                             length_ratio=hyp_len / ref_len, hyps=hyps,
                             params=this_params, decode_kw=this_decode_kw)

    print(f"\n=== qualitative examples (first {show_examples}) ===")
    for i in range(min(show_examples, len(src_raw))):
        print(f"\nSource: {src_raw[i]}")
        print(f"Target: {refs_joined[i]}")
        for name in lms:
            print(f"{name:>10}: {results[name]['hyps'][i]}")

    return results
