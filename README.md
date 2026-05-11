# AI-Powered Sentiment Tracking and Support System

This project packages the capstone thesis prototype into a self-contained local web application for emotional well-being reflection.

## What the system does

- Accepts short journal-style text entries.
- Estimates emotion probabilities across `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`, and `neutral`.
- Returns multi-label `active_labels` using a documented threshold, so one entry can show mixed emotions.
- Shows sentence-level emotion breakdown.
- Computes an `Emotional Health Index (EHI)` from 0 to 100.
- Stores trend history only in browser local storage.
- Works fully offline with a built-in Python analyzer and does not require cloud inference.
- Can optionally load a local HuggingFace-compatible DistilBERT/RoBERTa-style model through `SENTIMENT_MODEL_DIR`.
- Includes evaluation, usability, and privacy documentation to align the code with the thesis/poster claims.

## Project structure

- `app.py`: local backend server. Uses Flask when available, otherwise falls back to Python's built-in HTTP server.
- `model.py`: offline emotion analysis engine plus optional local transformer wrapper.
- `frontend/index.html`: single-page dashboard UI.
- `frontend/sw.js`: service worker for offline caching.
- `evaluate.py`: tiny demo evaluation script.
- `test_model.py`: unit tests for the analyzer.
- `generate_figures.py`: generates the thesis-facing prototype figures and evidence files.
- `data/`: label harmonization documentation and preprocessing notes.
- `scripts/`: preprocessing, transformer training, and transformer evaluation scripts.
- `evaluation/`: SUS pilot format, analysis script, and evaluation plan.
- `models/`: instructions for optional local model artifacts.
- `figures/`: generated prototype figures and supporting metrics.
- `docs/`: thesis and SRD poster PDFs.
- `reference/`: numbered bibliography/reference materials.
- `privacy_threat_model.md`: privacy and safety risk analysis.

## How to run

### Option 1: no dependencies

```bash
git clone https://github.com/shenyim/sentiment-web.git
cd sentiment-web
python3 app.py
```

Then open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

### Option 2: with Flask

```bash
cd sentiment-web
python3 -m pip install -r requirements.txt
python3 app.py
```

## How to test

```bash
cd sentiment-web
python3 -m unittest test_model.py
python3 evaluate.py
```

## Optional transformer path

The submitted demo uses the offline lexicon analyzer so it runs without downloads. If you need the code to match the DistilBERT/RoBERTa wording in the thesis more directly, install the optional dependencies and train or place a local HuggingFace-compatible model:

```bash
cd sentiment-web
python3 -m pip install -r requirements-transformer.txt
python3 scripts/preprocess.py --input data/raw/goemotions.csv --source goemotions --output data/processed_goemotions.jsonl
python3 scripts/train_transformer.py --train data/train.jsonl --validation data/validation.jsonl --output-dir models/distilbert-emotion
SENTIMENT_MODEL_DIR=models/distilbert-emotion python3 app.py
```

The API response format stays stable across analyzers, including `probs`, `active_labels`, `sentences`, `ehi`, `model_name`, and `model_type`.

## Evaluation and thesis evidence

- `scripts/evaluate_transformer.py` computes subset accuracy, hamming loss, micro-F1, macro-F1, and per-label metrics for a local transformer model.
- `generate_figures.py` still supports demo figures for the offline analyzer. These are grounded in the implemented analyzer, but they should be described as prototype/surrogate evidence unless replaced by transformer training logs.
- `evaluation/sus_analysis.py` computes SUS scores from `evaluation/sus_responses.csv`.
- `evaluation/evaluation_plan.md` lists the exact model and usability metrics to report.
- `privacy_threat_model.md` documents local storage, no-cloud inference, safety limitations, and future privacy improvements.

## How to generate the three thesis figures

Install dependencies first:

```bash
cd sentiment-web
python3 -m pip install -r requirements.txt
```

Then run:

```bash
python3 generate_figures.py
```

This will generate or refresh three PNG files in `figures/`:

- `figure_4_1_f1_radar.png`
- `figure_4_2_loss_curve.png`
- `figure_4_3_confusion_matrix.png`

It will also generate supporting evidence files:

- `figure_metrics.json`
- `benchmark_dataset.json`
- `per_label_metrics.csv`
- `threshold_sweep.csv`
- `thesis_alignment_report.md`

### Why these figures are logically grounded

- The benchmark labels are aligned with the project’s active emotion categories in `model.py`: `joy`, `sadness`, `anger`, `fear`, `love`, and `surprise`.
- The evaluation texts are not arbitrary placeholders. They are built from:
  - `EMOTION_LEXICON` in `model.py`
  - `PHRASE_HINTS` in `model.py`
  - the seed examples from `evaluate.py`
  - additional mixed-emotion journaling examples to better reflect the thesis emphasis on diary-style multi-label input
- The F1 radar chart and normalized confusion matrix are computed from the actual predictions produced by `create_analyzer()`.
- The script also computes multi-label metrics with threshold sweeps, including:
  - macro precision / recall / F1
  - micro precision / recall / F1
  - subset accuracy
  - hamming loss
- The loss curve is a reproducible surrogate training/validation curve based on staged model refinement:
  - gradually increasing lexicon coverage
  - then enabling phrase hints
  - then negation handling
  - then intensifier handling
  - then short-text bias and contrast cues
- This design matches the thesis context better than hard-coded demo numbers because the figures are now derived from the implemented analyzer logic and documented project assets.

## Notes for thesis submission

- The packaged version is privacy-first and local-only.
- The bundled analyzer is an offline heuristic model so the project can run without downloading external transformer weights.
- If you use the thesis wording that says the final system uses DistilBERT/RoBERTa, provide the trained model directory, transformer metrics, and training logs generated by the optional scripts.
- If you submit the current lightweight package only, describe the classifier as an offline heuristic analyzer and present DistilBERT/RoBERTa as an extension path or reference experiment.
- This prototype is not a clinical diagnostic system.
