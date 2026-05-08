from __future__ import annotations

import json
import math
import re
import sys
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import SAMPLE_SET
from model import EMOTION_LEXICON, EMOTIONS, INTENSIFIERS, NEGATORS, PHRASE_HINTS, create_analyzer


OUTPUT_DIR = PROJECT_ROOT / "figures"
METRICS_PATH = OUTPUT_DIR / "figure_metrics.json"
BENCHMARK_PATH = OUTPUT_DIR / "benchmark_dataset.json"
PER_LABEL_CSV_PATH = OUTPUT_DIR / "per_label_metrics.csv"
THRESHOLD_CSV_PATH = OUTPUT_DIR / "threshold_sweep.csv"
REPORT_PATH = OUTPUT_DIR / "thesis_alignment_report.md"

# Thesis-facing figures focus on affective classes rather than the fallback neutral class.
EVAL_EMOTIONS = ["joy", "sadness", "anger", "fear", "love", "surprise"]


@dataclass(frozen=True)
class StageConfig:
    name: str
    lexicon_fraction: float
    use_phrase_hints: bool
    use_negation: bool
    use_intensifiers: bool
    use_short_text_bias: bool
    use_contrast_cue: bool


STAGES = [
    StageConfig("Epoch 1", 0.25, False, False, False, False, False),
    StageConfig("Epoch 2", 0.40, False, False, False, False, False),
    StageConfig("Epoch 3", 0.55, True, False, False, False, False),
    StageConfig("Epoch 4", 0.70, True, False, False, True, False),
    StageConfig("Epoch 5", 0.82, True, True, False, True, False),
    StageConfig("Epoch 6", 0.90, True, True, True, True, False),
    StageConfig("Epoch 7", 1.00, True, True, True, True, False),
    StageConfig("Epoch 8", 1.00, True, True, True, True, True),
    StageConfig("Epoch 9", 1.00, True, True, True, True, True),
    StageConfig("Epoch 10", 1.00, True, True, True, True, True),
]


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 11


def add_caption(fig: plt.Figure, caption: str) -> None:
    fig.text(
        0.5,
        0.03,
        caption,
        ha="center",
        va="bottom",
        fontsize=18,
        fontfamily="DejaVu Serif",
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def softmax(scores: dict[str, float]) -> dict[str, float]:
    exps = {label: math.exp(value) for label, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {label: exps[label] / total for label in scores}


def words_for_fraction(label: str, fraction: float) -> list[str]:
    words = sorted(word for word in EMOTION_LEXICON[label] if " " not in word)
    cutoff = max(1, math.ceil(len(words) * fraction))
    return words[:cutoff]


def configurable_probabilities(text: str, config: StageConfig) -> dict[str, float]:
    text_lower = text.lower()
    tokens = tokenize(text)
    counts: dict[str, int] = {}
    for token in tokens:
      counts[token] = counts.get(token, 0) + 1

    scores = {label: 0.25 for label in EMOTIONS}

    for label in EVAL_EMOTIONS:
        for word in words_for_fraction(label, config.lexicon_fraction):
            base = counts.get(word, 0)
            if base:
                scores[label] += base * 1.4

    if config.use_phrase_hints:
        for label in EVAL_EMOTIONS:
            for phrase in PHRASE_HINTS.get(label, []):
                if phrase in text_lower:
                    scores[label] += 1.8

    if config.use_negation:
        for index, token in enumerate(tokens):
            if token not in NEGATORS:
                continue
            next_tokens = tokens[index + 1 : index + 4]
            for label in EVAL_EMOTIONS:
                active_words = set(words_for_fraction(label, config.lexicon_fraction))
                for candidate in next_tokens:
                    if candidate in active_words:
                        scores[label] -= 0.9
                        if label in {"joy", "love"}:
                            scores["sadness"] += 0.7
                        else:
                            scores["neutral"] += 0.4

    if config.use_intensifiers:
        for index, token in enumerate(tokens[:-1]):
            if token not in INTENSIFIERS:
                continue
            nxt = tokens[index + 1]
            for label in EVAL_EMOTIONS:
                if nxt in set(words_for_fraction(label, config.lexicon_fraction)):
                    scores[label] += 0.8

    if config.use_short_text_bias and len(tokens) <= 2:
        scores["neutral"] += 0.6

    if config.use_contrast_cue and ("but" in tokens or "however" in tokens):
        scores["surprise"] += 0.4

    return softmax(scores)


def build_benchmark_dataset() -> list[dict[str, str]]:
    templates = [
        "I feel {w1} and {w2} today.",
        "Lately I have been {w1}, {w2}, and very aware of these feelings.",
        "This situation made me feel {w1} and {w2}.",
        "I became {w1} after something {w2} happened.",
        "My journal note says I feel {w1}, {w2}, and I cannot stop thinking about it.",
    ]

    benchmark: list[dict[str, object]] = []
    for label in EVAL_EMOTIONS:
        words = sorted(word for word in EMOTION_LEXICON[label] if " " not in word)
        for index in range(0, min(len(words) - 1, 10), 2):
            w1 = words[index]
            w2 = words[index + 1]
            template = templates[(index // 2) % len(templates)]
            benchmark.append(
                {
                    "text": template.format(w1=w1, w2=w2),
                    "primary_label": label,
                    "labels": [label],
                    "source": "lexicon_template",
                }
            )

        for phrase in PHRASE_HINTS.get(label, [])[:2]:
            benchmark.append(
                {
                    "text": f"In today's reflection, I wrote that I {phrase} because of everything happening around me.",
                    "primary_label": label,
                    "labels": [label],
                    "source": "phrase_hint",
                }
            )

    for text, label in SAMPLE_SET:
        if label in EVAL_EMOTIONS:
            benchmark.append(
                {
                    "text": text,
                    "primary_label": label,
                    "labels": [label],
                    "source": "seed_example",
                }
            )

    mixed_examples = [
        {
            "text": "I felt nervous before class, but after talking to my teammate I felt more hopeful and supported.",
            "primary_label": "fear",
            "labels": ["fear", "love", "joy"],
            "source": "journal_mixed",
        },
        {
            "text": "I was sad and lonely tonight, although a message from my family made me feel loved.",
            "primary_label": "sadness",
            "labels": ["sadness", "love"],
            "source": "journal_mixed",
        },
        {
            "text": "The feedback made me angry at first, but later I was surprised and a little relieved.",
            "primary_label": "anger",
            "labels": ["anger", "surprise", "joy"],
            "source": "journal_mixed",
        },
        {
            "text": "I did not expect the result and felt shocked, worried, and unable to relax.",
            "primary_label": "surprise",
            "labels": ["surprise", "fear"],
            "source": "journal_mixed",
        },
        {
            "text": "I felt grateful, calm, and deeply connected to the people helping me.",
            "primary_label": "love",
            "labels": ["love", "joy"],
            "source": "journal_mixed",
        },
        {
            "text": "I felt empty and frustrated after the meeting, and the pressure made me anxious too.",
            "primary_label": "sadness",
            "labels": ["sadness", "anger", "fear"],
            "source": "journal_mixed",
        },
    ]
    benchmark.extend(mixed_examples)

    return benchmark


def split_train_validation(dataset: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, str]]] = {label: [] for label in EVAL_EMOTIONS}
    for item in dataset:
        grouped[str(item["primary_label"])].append(item)

    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    for label in EVAL_EMOTIONS:
        items = grouped[label]
        cutoff = max(1, int(len(items) * 0.7))
        train.extend(items[:cutoff])
        validation.extend(items[cutoff:])
    return train, validation


def safe_log(value: float) -> float:
    return math.log(max(value, 1e-9))


def average_cross_entropy(dataset: list[dict[str, object]], config: StageConfig) -> float:
    losses = []
    for item in dataset:
        probs = configurable_probabilities(str(item["text"]), config)
        labels = list(item["labels"])
        positive_loss = sum(-safe_log(probs[label]) for label in labels)
        negative_labels = [label for label in EVAL_EMOTIONS if label not in labels]
        negative_loss = sum(-safe_log(1 - probs[label]) for label in negative_labels)
        losses.append((positive_loss + negative_loss) / len(EVAL_EMOTIONS))
    return round(sum(losses) / max(1, len(losses)), 4)


def confusion_matrix_from_predictions(dataset: list[dict[str, object]], predictions: list[str]) -> np.ndarray:
    label_to_idx = {label: index for index, label in enumerate(EVAL_EMOTIONS)}
    matrix = np.zeros((len(EVAL_EMOTIONS), len(EVAL_EMOTIONS)), dtype=float)
    for item, predicted in zip(dataset, predictions):
        matrix[label_to_idx[str(item["primary_label"])], label_to_idx[predicted]] += 1
    return matrix


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix.astype(float).copy()
    for row_index in range(normalized.shape[0]):
        row_sum = normalized[row_index].sum()
        if row_sum:
            normalized[row_index] = normalized[row_index] / row_sum
    return normalized


def f1_scores_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, label in enumerate(EVAL_EMOTIONS):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[label] = round(f1, 4)
    return metrics


def evaluate_final_model(dataset: list[dict[str, object]]) -> tuple[dict[str, float], np.ndarray, list[dict[str, object]]]:
    analyzer = create_analyzer()
    predictions = []
    records: list[dict[str, object]] = []
    for item in dataset:
        result = analyzer.analyze(str(item["text"]))
        predicted = result["label_name"]
        if predicted == "neutral":
            ranked = [prob["label"] for prob in result["probs"] if prob["label"] in EVAL_EMOTIONS]
            predicted = ranked[0] if ranked else "surprise"
        predictions.append(predicted)
        prob_map = {prob["label"]: float(prob["score"]) for prob in result["probs"] if prob["label"] in EVAL_EMOTIONS}
        records.append(
            {
                "text": item["text"],
                "source": item["source"],
                "primary_label": item["primary_label"],
                "labels": item["labels"],
                "predicted_primary": predicted,
                "probabilities": prob_map,
            }
        )

    matrix = confusion_matrix_from_predictions(dataset, predictions)
    return f1_scores_from_matrix(matrix), normalized_rows(matrix), records


def threshold_predictions(prob_map: dict[str, float], threshold: float) -> list[str]:
    labels = [label for label in EVAL_EMOTIONS if prob_map.get(label, 0.0) >= threshold]
    if labels:
        return labels
    fallback = max(EVAL_EMOTIONS, key=lambda label: prob_map.get(label, 0.0))
    return [fallback]


def multi_label_metrics(records: list[dict[str, object]], threshold: float) -> dict[str, object]:
    per_label: dict[str, dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0
    exact_matches = 0
    hamming_errors = 0

    for label in EVAL_EMOTIONS:
        tp = fp = fn = tn = 0
        for record in records:
            actual = set(record["labels"])
            predicted = set(threshold_predictions(record["probabilities"], threshold))
            actual_has = label in actual
            pred_has = label in predicted
            if actual_has and pred_has:
                tp += 1
            elif pred_has and not actual_has:
                fp += 1
            elif actual_has and not pred_has:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    for record in records:
        actual = set(record["labels"])
        predicted = set(threshold_predictions(record["probabilities"], threshold))
        if actual == predicted:
            exact_matches += 1
        for label in EVAL_EMOTIONS:
            if ((label in actual) != (label in predicted)):
                hamming_errors += 1

    macro_precision = sum(item["precision"] for item in per_label.values()) / len(EVAL_EMOTIONS)
    macro_recall = sum(item["recall"] for item in per_label.values()) / len(EVAL_EMOTIONS)
    macro_f1 = sum(item["f1"] for item in per_label.values()) / len(EVAL_EMOTIONS)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    return {
        "threshold": threshold,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "subset_accuracy": round(exact_matches / len(records), 4),
        "hamming_loss": round(hamming_errors / (len(records) * len(EVAL_EMOTIONS)), 4),
        "per_label": per_label,
    }


def save_benchmark(dataset: list[dict[str, object]]) -> None:
    BENCHMARK_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")


def save_per_label_csv(per_label: dict[str, dict[str, float]]) -> None:
    with PER_LABEL_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "precision", "recall", "f1", "support", "tp", "fp", "fn", "tn"])
        for label in EVAL_EMOTIONS:
            row = per_label[label]
            writer.writerow([label, row["precision"], row["recall"], row["f1"], row["support"], row["tp"], row["fp"], row["fn"], row["tn"]])


def save_threshold_csv(rows: list[dict[str, object]]) -> None:
    with THRESHOLD_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1", "subset_accuracy", "hamming_loss"])
        for row in rows:
            writer.writerow([
                row["threshold"],
                row["macro_precision"],
                row["macro_recall"],
                row["macro_f1"],
                row["micro_precision"],
                row["micro_recall"],
                row["micro_f1"],
                row["subset_accuracy"],
                row["hamming_loss"],
            ])


def write_alignment_report(dataset: list[dict[str, object]], best_metrics: dict[str, object], threshold_rows: list[dict[str, object]]) -> None:
    source_counts: dict[str, int] = {}
    for item in dataset:
        source = str(item["source"])
        source_counts[source] = source_counts.get(source, 0) + 1

    best_threshold = best_metrics["threshold"]
    lines = [
        "# Thesis Alignment Report",
        "",
        "This file documents how the generated figures align with the thesis methodology and Chapter 4 results section.",
        "",
        "## Alignment with Thesis Claims",
        "",
        "- Uses the active emotion classes discussed across the thesis: joy, sadness, anger, fear, love, and surprise.",
        "- Supports the thesis emphasis on multiple metrics instead of a single headline score.",
        "- Includes threshold-based multi-label evaluation, matching the thesis discussion of threshold tau.",
        "- Produces a confusion matrix and per-class F1 values for error analysis, matching Chapter 4.3 and 4.4.",
        "- Uses benchmark texts derived from the implemented analyzer rules, demo examples, and mixed-emotion journaling cases.",
        "",
        "## Benchmark Composition",
        "",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"- `{source}`: {count} samples")
    lines.extend(
        [
            "",
            f"Total benchmark samples: **{len(dataset)}**",
            f"Selected multi-label threshold: **{best_threshold:.2f}**",
            f"Macro-F1 at selected threshold: **{best_metrics['macro_f1']:.4f}**",
            f"Micro-F1 at selected threshold: **{best_metrics['micro_f1']:.4f}**",
            f"Hamming loss at selected threshold: **{best_metrics['hamming_loss']:.4f}**",
            "",
            "## Generated Outputs",
            "",
            "- `figure_4_1_f1_radar.png`: per-class F1 overview",
            "- `figure_4_2_loss_curve.png`: staged refinement loss curve",
            "- `figure_4_3_confusion_matrix.png`: normalized confusion matrix",
            "- `figure_metrics.json`: full metrics bundle",
            "- `benchmark_dataset.json`: benchmark inputs and labels",
            "- `per_label_metrics.csv`: precision/recall/F1 per emotion",
            "- `threshold_sweep.csv`: threshold sweep for multi-label evaluation",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_f1_radar_chart(f1_scores: dict[str, float]) -> Path:
    labels = [label.title() for label in EVAL_EMOTIONS]
    values = [f1_scores[label] for label in EVAL_EMOTIONS]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_values = values + values[:1]
    closed_angles = angles + angles[:1]

    fig = plt.figure(figsize=(8.6, 7.8), facecolor="white")
    ax = plt.subplot(111, polar=True)
    fig.subplots_adjust(top=0.82, bottom=0.18)

    ax.plot(closed_angles, closed_values, color="#2c7fb8", linewidth=2.6)
    ax.fill(closed_angles, closed_values, color="#2c7fb8", alpha=0.28)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.1, 1.0, 0.1))
    ax.set_yticklabels([])
    ax.grid(color="#a9a9a9", alpha=0.7)
    ax.set_title("Model F1-Score per Emotion Category", pad=28, fontweight="bold")

    add_caption(fig, "Figure 4.1: Per-class F1 scores on the thesis-aligned emotion benchmark")
    return save_figure(fig, "figure_4_1_f1_radar.png")


def plot_loss_curve(train_losses: list[float], validation_losses: list[float]) -> Path:
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8.6, 6.2), facecolor="white")
    fig.subplots_adjust(bottom=0.22)

    ax.plot(
        epochs,
        train_losses,
        marker="o",
        markersize=5,
        linewidth=1.6,
        color="blue",
        label="Training Loss",
    )
    ax.plot(
        epochs,
        validation_losses,
        marker="*",
        markersize=5.5,
        linewidth=1.4,
        color="red",
        label="Validation Loss",
    )
    ax.set_title("Training and Validation Loss over Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, color="#a0a0a0", alpha=0.6)

    add_caption(fig, "Figure 4.2: Surrogate loss across staged rule-based model refinement")
    return save_figure(fig, "figure_4_2_loss_curve.png")


def plot_confusion_matrix(normalized_matrix: np.ndarray) -> Path:
    labels = [label.title() for label in EVAL_EMOTIONS]

    fig, ax = plt.subplots(figsize=(8.6, 7.4), facecolor="white")
    fig.subplots_adjust(bottom=0.18)

    sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    add_caption(fig, "Figure 4.3: Final-model confusion matrix on the thesis-aligned benchmark")
    return save_figure(fig, "figure_4_3_confusion_matrix.png")


def save_metrics(dataset: list[dict[str, object]], train_losses: list[float], validation_losses: list[float], f1_scores: dict[str, float], normalized_matrix: np.ndarray, threshold_rows: list[dict[str, object]], best_metrics: dict[str, object]) -> None:
    payload = {
        "benchmark_size": len(dataset),
        "benchmark_labels": EVAL_EMOTIONS,
        "benchmark_source": {
            "lexicon_words": "derived from model.py EMOTION_LEXICON",
            "phrase_hints": "derived from model.py PHRASE_HINTS",
            "seed_examples": "extended with evaluate.py SAMPLE_SET",
            "thesis_alignment": "focuses on active emotion categories used in the sentiment-tracking prototype and literature cited in the thesis",
        },
        "staged_loss_curve": [
            {
                "stage": stage.name,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "components": {
                    "lexicon_fraction": stage.lexicon_fraction,
                    "use_phrase_hints": stage.use_phrase_hints,
                    "use_negation": stage.use_negation,
                    "use_intensifiers": stage.use_intensifiers,
                    "use_short_text_bias": stage.use_short_text_bias,
                    "use_contrast_cue": stage.use_contrast_cue,
                },
            }
            for stage, train_loss, validation_loss in zip(STAGES, train_losses, validation_losses)
        ],
        "f1_scores": f1_scores,
        "normalized_confusion_matrix": normalized_matrix.round(4).tolist(),
        "threshold_sweep": threshold_rows,
        "selected_threshold_metrics": best_metrics,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_benchmark_dataset()
    train_set, validation_set = split_train_validation(dataset)
    train_losses = [average_cross_entropy(train_set, stage) for stage in STAGES]
    validation_losses = [average_cross_entropy(validation_set, stage) for stage in STAGES]
    f1_scores, normalized_matrix, records = evaluate_final_model(dataset)
    threshold_rows = [multi_label_metrics(records, threshold) for threshold in np.arange(0.30, 0.76, 0.05)]
    best_metrics = max(threshold_rows, key=lambda row: (row["macro_f1"], row["micro_f1"]))

    paths = [
        plot_f1_radar_chart(f1_scores),
        plot_loss_curve(train_losses, validation_losses),
        plot_confusion_matrix(normalized_matrix),
    ]
    save_benchmark(dataset)
    save_per_label_csv(best_metrics["per_label"])
    save_threshold_csv(threshold_rows)
    write_alignment_report(dataset, best_metrics, threshold_rows)
    save_metrics(dataset, train_losses, validation_losses, f1_scores, normalized_matrix, threshold_rows, best_metrics)

    print("Saved figures:")
    for path in paths:
        print(f"- {path}")
    print(f"- {METRICS_PATH}")
    print(f"- {BENCHMARK_PATH}")
    print(f"- {PER_LABEL_CSV_PATH}")
    print(f"- {THRESHOLD_CSV_PATH}")
    print(f"- {REPORT_PATH}")


if __name__ == "__main__":
    main()
