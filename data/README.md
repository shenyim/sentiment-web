# Data Pipeline Notes

This folder documents the dataset alignment expected by the thesis and poster.

The submitted demo can run without external data because it includes an offline lexicon analyzer. For the transformer experiment path, place raw public dataset exports outside Git or in an ignored `data/raw/` folder, then run:

```bash
python3 scripts/preprocess.py --input data/raw/goemotions.csv --source goemotions --output data/processed_goemotions.jsonl
```

Each processed row is JSONL with:

- `text`: cleaned journal/social text
- `labels`: one or more labels from `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`, `neutral`
- `source`: dataset source name

The project does not ship public datasets or user journal text. This avoids redistributing third-party data and keeps the capstone package small.
