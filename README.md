# AI-Powered Sentiment Tracking and Support System

This repository packages the capstone thesis prototype into a self-contained local web application for emotional well-being reflection.

The system is privacy-first: it runs with a built-in Python analyzer, does not require cloud inference, and keeps diary history in the browser's local storage.

## What The System Does

- Accepts short journal-style text entries.
- Estimates emotion probabilities across `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`, and `neutral`.
- Shows sentence-level emotion breakdown.
- Computes an `Emotional Health Index (EHI)` from 0 to 100.
- Stores trend history only in browser local storage.
- Works offline with the bundled Python analyzer after the local server starts.

## Repository Structure

```text
.
├── app.py                  # Local backend server
├── model.py                # Offline emotion analysis engine
├── evaluate.py             # Demo evaluation script
├── test_model.py           # Unit tests for the analyzer
├── requirements.txt        # Optional Python dependencies
├── frontend/               # Browser dashboard UI and PWA files
├── scripts/                # Figure generation scripts
├── figures/                # Generated thesis figures and metric evidence
├── docs/                   # Thesis and poster PDFs
├── media/                  # Supporting visual assets
└── data/                   # Dataset notes and reconstruction guidance
```

## How To Run

### Option 1: no dependencies

```bash
python3 app.py
```

Then open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

### Option 2: with Flask

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

The Flask server and the built-in fallback server expose the same local endpoints:

- `GET /health`
- `POST /predict`

## How To Test

```bash
python3 -m unittest test_model.py
python3 evaluate.py
```

## How To Generate Thesis Figures

Install optional plotting dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

Then run:

```bash
python3 scripts/generate_figures.py
```

This generates thesis figures and supporting evidence files in `figures/`, including:

- `figure_4_1_f1_radar.png`
- `figure_4_2_loss_curve.png`
- `figure_4_3_confusion_matrix.png`
- `figure_metrics.json`
- `benchmark_dataset.json`
- `per_label_metrics.csv`
- `threshold_sweep.csv`
- `thesis_alignment_report.md`

## Thesis And Poster Materials

- Thesis PDF: [`docs/thesis.pdf`](docs/thesis.pdf)
- SRD poster PDF: [`docs/2026-SRD-Poster.pdf`](docs/2026-SRD-Poster.pdf)

## Data Notes

The repository does not include private user diary data. The analyzer is evaluated with reproducible benchmark texts derived from the implemented lexicon, phrase hints, seed examples, and mixed-emotion journaling examples. See [`data/README.md`](data/README.md) for details.

## Why The Figures Are Logically Grounded

- The benchmark labels are aligned with the project emotion categories in `model.py`.
- Evaluation texts are derived from `EMOTION_LEXICON`, `PHRASE_HINTS`, `evaluate.py`, and additional diary-style mixed-emotion examples.
- The F1 radar chart and normalized confusion matrix are computed from actual predictions produced by `create_analyzer()`.
- The script computes multi-label metrics with threshold sweeps, including macro precision/recall/F1, micro precision/recall/F1, subset accuracy, and hamming loss.
- The loss curve is a reproducible surrogate training/validation curve based on staged model refinement.

## Notes For Thesis Submission

- The packaged version is privacy-first and local-only.
- The bundled analyzer is an offline heuristic model, so the project can run without downloading external transformer weights.
- A future transformer model can replace the implementation inside `model.py` if it keeps the same API response format.
- This prototype is not a clinical diagnostic system.
