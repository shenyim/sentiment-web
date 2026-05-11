from __future__ import annotations

import csv
import json
from pathlib import Path


QUESTIONS = [f"q{index}" for index in range(1, 11)]
POSITIVE_ITEMS = {"q1", "q3", "q5", "q7", "q9"}


def sus_score(row: dict[str, str]) -> float:
    total = 0
    for question in QUESTIONS:
        value = int(row[question])
        if question in POSITIVE_ITEMS:
            total += value - 1
        else:
            total += 5 - value
    return total * 2.5


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    responses_path = base_dir / "sus_responses.csv"
    output_path = base_dir / "sus_summary.json"
    rows = list(csv.DictReader(responses_path.open("r", encoding="utf-8")))
    scores = [sus_score(row) for row in rows]
    summary = {
        "participants": len(scores),
        "scores": scores,
        "mean_sus": round(sum(scores) / len(scores), 2) if scores else 0,
        "min_sus": min(scores) if scores else 0,
        "max_sus": max(scores) if scores else 0,
        "interpretation": "Prototype usability evidence for capstone reporting; replace with final study responses before submission.",
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
