# Thesis Alignment Report

This file documents how the generated figures align with the thesis methodology and Chapter 4 results section.

## Alignment with Thesis Claims

- Uses the active emotion classes discussed across the thesis: joy, sadness, anger, fear, love, and surprise.
- Supports the thesis emphasis on multiple metrics instead of a single headline score.
- Includes threshold-based multi-label evaluation, matching the thesis discussion of threshold tau.
- Produces a confusion matrix and per-class F1 values for error analysis, matching Chapter 4.3 and 4.4.
- Uses benchmark texts derived from the implemented analyzer rules, demo examples, and mixed-emotion journaling cases.

## Benchmark Composition

- `journal_mixed`: 6 samples
- `lexicon_template`: 30 samples
- `phrase_hint`: 12 samples
- `seed_example`: 6 samples

Total benchmark samples: **54**
Selected multi-label threshold: **0.30**
Macro-F1 at selected threshold: **0.9217**
Micro-F1 at selected threshold: **0.9231**
Hamming loss at selected threshold: **0.0278**

## Generated Outputs

- `figure_4_1_f1_radar.png`: per-class F1 overview
- `figure_4_2_loss_curve.png`: staged refinement loss curve
- `figure_4_3_confusion_matrix.png`: normalized confusion matrix
- `figure_metrics.json`: full metrics bundle
- `benchmark_dataset.json`: benchmark inputs and labels
- `per_label_metrics.csv`: precision/recall/F1 per emotion
- `threshold_sweep.csv`: threshold sweep for multi-label evaluation
