from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LABELS = ["joy", "sadness", "anger", "fear", "love", "surprise", "neutral"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def f1_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local transformer model on processed JSONL data.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", default=Path("evaluation/transformer_metrics.json"), type=Path)
    parser.add_argument("--per-label-csv", default=Path("evaluation/transformer_per_label.csv"), type=Path)
    parser.add_argument("--threshold", default=0.18, type=float)
    args = parser.parse_args()

    try:
        from model import TransformerEmotionAnalyzer
    except ImportError as exc:
        raise SystemExit("Run this script from the project root.") from exc

    analyzer = TransformerEmotionAnalyzer(args.model_dir, threshold=args.threshold)
    rows = load_jsonl(args.data)
    per_label = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    exact = 0
    total_decisions = 0
    wrong_decisions = 0

    for row in rows:
        expected = set(row["labels"])
        result = analyzer.analyze(row["text"])
        predicted = {item["label"] for item in result["active_labels"]}
        if predicted == expected:
            exact += 1
        for label in LABELS:
            in_expected = label in expected
            in_predicted = label in predicted
            if in_expected and in_predicted:
                per_label[label]["tp"] += 1
            elif not in_expected and in_predicted:
                per_label[label]["fp"] += 1
                wrong_decisions += 1
            elif in_expected and not in_predicted:
                per_label[label]["fn"] += 1
                wrong_decisions += 1
            total_decisions += 1

    label_metrics = {}
    micro_counts = {"tp": 0, "fp": 0, "fn": 0}
    for label, counts in per_label.items():
        precision, recall, f1 = f1_counts(counts["tp"], counts["fp"], counts["fn"])
        label_metrics[label] = {"precision": precision, "recall": recall, "f1": f1, **counts}
        for key in micro_counts:
            micro_counts[key] += counts[key]

    micro = f1_counts(micro_counts["tp"], micro_counts["fp"], micro_counts["fn"])
    macro_f1 = sum(item["f1"] for item in label_metrics.values()) / len(label_metrics)
    metrics = {
        "threshold": args.threshold,
        "examples": len(rows),
        "subset_accuracy": exact / len(rows) if rows else 0.0,
        "hamming_loss": wrong_decisions / total_decisions if total_decisions else 0.0,
        "micro_precision": micro[0],
        "micro_recall": micro[1],
        "micro_f1": micro[2],
        "macro_f1": macro_f1,
        "per_label": label_metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.per_label_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.per_label_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["label", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        for label, values in label_metrics.items():
            writer.writerow({"label": label, **values})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
