# Noisy Channel Approach on Europarl for Machine Translation

A statistical machine translation project built from Europarl IT–EN: word alignment, translation tables, a beam-search decoder, with pluggable language models. It asks one question: **how far can a better language model take a noisy-channel translator, and what limits it?**

The system is compared against a seq2seq model trained from scratch and against the pretrained `opus-mt-it-en`, which serves as an upper bound.

## How it works

Translation is factored with Bayes' rule: `ê = argmax_e P(e) · P(f | e)`. A translation model checks fidelity, and a language model checks fluency.

1. **Data preparation:** Europarl IT–EN pairs of 4–20 words per side with a length ratio of at most 2.5 (about 600k pairs), downsampled to 200k (seed 0). Italian uses a regex tokenizer and English uses NLTK, both lowercased. Subword variants use the SentencePiece models from `Helsinki-NLP/opus-mt-it-en`.

2. **Alignment:** [eflomal](https://github.com/robertostling/eflomal) (IBM 1 → HMM with fertility) runs in both directions, and the links are intersected.

3. **Translation tables:** pruned `P(e | f)` and `P(f | e)` are built from the intersected links. Source words with no entry are copied through, and words aligned to nothing are deleted.

4. **Decoder:** a beam search that, at each step, picks an uncovered source word and emits one target word or deletes it. The log-linear score combines the language model, the two table probabilities, a word bonus and a distortion penalty. Weights are tuned by grid search on a dev set. There are no insertions and no phrases.

5. **Language models** plug into the decoder: Kneser–Ney n-grams (orders 1–3), an LSTM LM, and Gemma-3-1B (`google/gemma-3-1b-pt`), each at word or SentencePiece level.

## Repository structure

```
.
├── docs/                  # slides (Quarto reveal.js)
│   ├── mt_slides.qmd      # slide source
│   ├── MT.css             # slide theme
│   └── plots/
├── utils/
│   ├── main_pipeline.py   # data prep, tokenization, eflomal, tables, decoder, tuning
│   ├── ngram.py           # Kneser–Ney n-gram LM
│   ├── LSTMLM.py          # LSTM language model
│   ├── gemma.py           # Gemma wordLM
│   ├── lstmSeq2Seq.py     # seq2seq baseline
│   ├── opus_mt_test.py    # opus-mt evaluation
│   └── comparison.py      # evaluation of all systems on the same sentences
├── pyproject.toml
└── README.md
```

## Setup

The repo has a `pyproject.toml` file for uv users

```bash
uv sync
```
And also provides `requirements.txt` for standard pip usage

```bash
python3 -m venv .venv
source .venv/bin/activate       # MacOS, Linux, WSL
.venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
```
PyTorch is required but not present in pyproject.toml/requirements.txt. Install the appropriate wheel from the [official website](https://pytorch.org/get-started/locally/). For reference, this project was written using `torch==2.11.0+cu126`.

## Data

The pipeline expects a CSV with columns `sent_it` and `sent_en` (Europarl Italian–English sentence pairs).

`prepare_data` writes the tokenized parallel files `train.it` / `train.en` (and `.sp` subword versions). The following are generated, but kept out of git:

- `forward.align`, `reverse.align`: eflomal links
- `tables.pkl`: pruned translation tables
- `*.pt`: trained LSTM and seq2seq checkpoints

## Usage

The steps, in order: prepare data, run eflomal, build and save the tables, train a language model, tune the decoder weights, then evaluate with `comparison.py`. Models are evaluated on 200 sampled sentences (`val_df`) and on a set of about 100 short sentences (`val_short`), with BLEU and chrF from `sacrebleu`. Seed used: 0

## Results

`val_df` (100 sentences). Perplexity is per word and on the reference sentences. The decoder uses beam 10 and 5 candidates per word.

| System | Perplexity | BLEU | chrF |
|---|---|---|---|
| Noisy channel + unigram | 615.3 | 2.5 | 43.5 |
| Noisy channel + bigram | 101.8 | 13.3 | 49.8 |
| **Noisy channel + trigram** | 61.3 | **15.5** | 50.4 |
| Noisy channel + LSTM | 29.4 | 12.1 | 49.0 |
| Noisy channel + Gemma-3-1B | 36.0 | 13.3 | 54.4 |
| Seq2seq from scratch | n/a | 10.3 | 31.7 |
| opus-mt-it-en | n/a | 32.9 | 57.7 |

**Findings**

- BLEU rises with n-gram order (unigram 2.5, bigram 13.3, trigram 15.5) and then stalls. LMs with half the trigram's perplexity (LSTM, Gemma) do not translate better.
- SentencePiece units help the LSTM (12.1 → 14.0 BLEU) but hurt the trigram (15.5 → 12.9).
- A wider beam helps slightly (SentencePiece trigram: 12.9 → 14.4 BLEU going from beam 10 to beam 50, at 11× the time), so the main limit is the word-by-word model, not the search.

## Limitations

- Small evaluation sets (about 100 and 200 sentences) and single runs, so differences under about 1–2 BLEU may be noise.
- The seq2seq baseline produces outputs about half the reference length (length ratio 0.50), so it is probably undertrained.
- opus-mt is trained on far more data, including Europarl, so it is a ceiling, not a fair competitor.

## Slides

The slides in `docs/` use [Quarto](https://quarto.org) reveal.js. Render them with:

```bash
quarto render docs/mt_slides.qmd
```

## Credits

[eflomal](https://github.com/robertostling/eflomal), [Helsinki-NLP/opus-mt-it-en](https://huggingface.co/Helsinki-NLP/opus-mt-it-en), [Gemma 3](https://huggingface.co/google/gemma-3-1b-pt), [SentencePiece](https://github.com/google/sentencepiece), [sacreBLEU](https://github.com/mjpost/sacrebleu), Europarl.