from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
LABEL_MAP_PATH = BASE_DIR / "data" / "label_map.json"


def clean_text(value: str) -> str:
    value = re.sub(r"https?://\S+", "", value or "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_label_map() -> dict:
    return json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))


def parse_labels(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in re.split(r"[,;|]", raw) if item.strip()]


def convert_row(row: dict[str, str], source: str, mapping: dict[str, str]) -> dict[str, object] | None:
    text = clean_text(row.get("text") or row.get("comment_text") or row.get("sentence") or "")
    labels = parse_labels(row.get("labels") or row.get("label") or row.get("emotion") or "")
    mapped = sorted({mapping[label] for label in labels if label in mapping})
    if not text or not mapped:
        return None
    return {"text": text, "labels": mapped, "source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize public emotion datasets into thesis label space.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["goemotions", "emotion_dataset"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = load_label_map()
    mapping = config["source_mappings"][args.source]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with args.input.open("r", encoding="utf-8", newline="") as input_file, args.output.open("w", encoding="utf-8") as output_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            converted = convert_row(row, args.source, mapping)
            if converted is None:
                skipped += 1
                continue
            output_file.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1

    print(json.dumps({"kept": kept, "skipped": skipped, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
