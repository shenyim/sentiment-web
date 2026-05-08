from __future__ import annotations

from collections import Counter

from model import create_analyzer


SAMPLE_SET = [
    ("I am happy and excited about my progress.", "joy"),
    ("I feel lonely, tired, and empty tonight.", "sadness"),
    ("I am furious about how unfair this situation is.", "anger"),
    ("I am anxious and stressed about tomorrow's exam.", "fear"),
    ("I feel loved and supported by my friends.", "love"),
    ("I was shocked by the sudden news this morning.", "surprise"),
    ("Today felt stable and fairly ordinary overall.", "neutral"),
]


def main() -> None:
    analyzer = create_analyzer()
    correct = 0
    predictions = Counter()

    print("Sample evaluation")
    print("-" * 60)
    for text, expected in SAMPLE_SET:
        result = analyzer.analyze(text)
        predicted = result["label_name"]
        predictions[predicted] += 1
        hit = predicted == expected
        correct += int(hit)
        print(f"expected={expected:<8} predicted={predicted:<8} ehi={result['ehi']:<5} text={text}")

    accuracy = correct / len(SAMPLE_SET)
    print("-" * 60)
    print(f"Accuracy on bundled demo set: {accuracy:.2%}")
    print(f"Prediction distribution: {dict(predictions)}")
    print("Note: this is a tiny demonstration dataset for project packaging, not a thesis-grade benchmark.")


if __name__ == "__main__":
    main()
