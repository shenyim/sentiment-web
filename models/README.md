# Model Artifacts

The default demo uses `offline-lexicon-analyzer` from `model.py`, so no downloaded model is required.

To run the optional transformer implementation:

1. Fine-tune a model with `scripts/train_transformer.py`, or place an existing HuggingFace-compatible model folder here.
2. Start the app with:

```bash
SENTIMENT_MODEL_DIR=models/distilbert-emotion python3 app.py
```

The API response format stays the same for both analyzer types:

- `probs`: full emotion probability ranking
- `active_labels`: multi-label threshold output
- `ehi`: Emotional Health Index
- `sentences`: sentence-level analysis
- `model_name` and `model_type`: model provenance

Large model weights should normally stay out of Git and be documented in this folder instead.
