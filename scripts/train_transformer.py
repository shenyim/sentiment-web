from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = ["joy", "sadness", "anger", "fear", "love", "surprise", "neutral"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a local multi-label DistilBERT emotion model.")
    parser.add_argument("--train", required=True, type=Path, help="Processed JSONL from scripts/preprocess.py")
    parser.add_argument("--validation", required=True, type=Path, help="Processed JSONL validation split")
    parser.add_argument("--output-dir", default="models/distilbert-emotion", type=Path)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    args = parser.parse_args()

    try:
        import numpy as np
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing transformer training dependencies. Install them with: "
            "python3 -m pip install -r requirements-transformer.txt"
        ) from exc

    label_to_id = {label: index for index, label in enumerate(LABELS)}

    def encode_labels(labels: list[str]) -> list[float]:
        vector = [0.0] * len(LABELS)
        for label in labels:
            if label in label_to_id:
                vector[label_to_id[label]] = 1.0
        return vector

    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def to_dataset(rows: list[dict]) -> Dataset:
        dataset = Dataset.from_list(
            [{"text": row["text"], "labels": encode_labels(row["labels"])} for row in rows]
        )
        return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True), batched=True)

    train_dataset = to_dataset(train_rows)
    validation_dataset = to_dataset(validation_rows)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps({key: float(value) if isinstance(value, np.floating) else value for key, value in metrics.items()}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"model_dir": str(args.output_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
