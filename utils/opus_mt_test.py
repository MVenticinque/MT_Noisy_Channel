import time, torch, sacrebleu
from transformers import MarianMTModel, MarianTokenizer

mname = "Helsinki-NLP/opus-mt-it-en"
tok = MarianMTModel and MarianTokenizer.from_pretrained(mname)
model = MarianMTModel.from_pretrained(mname).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@torch.no_grad()
def opus_translate(texts, beam=5, batch_size=16, max_new_tokens=80):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i+batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=128).to(device)
        gen = model.generate(**batch, num_beams=beam, max_new_tokens=max_new_tokens)
        outs += tok.batch_decode(gen, skip_special_tokens=True)
    return outs

def eval_opus(val_df, n_test=200, seed=0, beam=5):
    df = val_df.dropna(subset=["sent_it", "sent_en"]).sample(
        n=min(n_test, len(val_df)), random_state=seed)
    src_raw = df["sent_it"].tolist()
    refs = [" ".join(tok_en(s)) for s in df["sent_en"]]

    t0 = time.time()
    hyps_raw = opus_translate(src_raw, beam=beam)
    elapsed = time.time() - t0

    hyps = [" ".join(tok_en(h)) for h in hyps_raw]
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    lr = sum(len(h.split()) for h in hyps) / sum(len(r.split()) for r in refs)
    print(f"opus-mt: BLEU {bleu:.2f}  chrF {chrf:.2f}  len-ratio {lr:.2f}  "
          f"{elapsed:.1f}s ({elapsed/len(src_raw)*1000:.0f}ms/sent)")
    return dict(bleu=bleu, chrf=chrf, time_s=elapsed, hyps=hyps, hyps_raw=hyps_raw)
