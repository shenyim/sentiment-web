# Evaluation Plan

The project has two evaluation tracks.

## 1. Model Evaluation

For the shipped offline analyzer:

```bash
python3 -m unittest test_model.py
python3 evaluate.py
python3 generate_figures.py
```

For the optional DistilBERT/RoBERTa track:

```bash
python3 scripts/preprocess.py --input data/raw/goemotions.csv --source goemotions --output data/processed_goemotions.jsonl
python3 scripts/train_transformer.py --train data/train.jsonl --validation data/validation.jsonl --output-dir models/distilbert-emotion
SENTIMENT_MODEL_DIR=models/distilbert-emotion python3 scripts/evaluate_transformer.py --model-dir models/distilbert-emotion --data data/test.jsonl
```

Metrics to report:

- subset accuracy
- hamming loss
- macro precision / recall / F1
- micro precision / recall / F1
- per-label precision / recall / F1
- confusion/error analysis examples

## 2. Usability Evaluation

Use the SUS questionnaire plus a short reflection prompt after participants try:

- writing a sample journal entry
- interpreting emotion probabilities
- reviewing the EHI trend
- clearing or exporting local history

The analysis script computes SUS scores from `sus_responses.csv`.
