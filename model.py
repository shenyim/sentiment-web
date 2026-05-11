from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


EMOTIONS = ["joy", "sadness", "anger", "fear", "love", "surprise", "neutral"]
DEFAULT_MULTI_LABEL_THRESHOLD = 0.18

EMOTION_LEXICON = {
    "joy": {
        "happy", "calm", "relieved", "optimistic", "hopeful", "good", "better",
        "grateful", "thankful", "peaceful", "confident", "proud", "excited",
        "motivated", "smile", "joy", "content", "fine", "great", "bright",
    },
    "sadness": {
        "sad", "down", "upset", "depressed", "cry", "crying", "lonely", "empty",
        "tired", "hopeless", "hurt", "heartbroken", "miserable", "drained",
        "guilty", "ashamed", "lost", "blue", "unhappy", "low",
    },
    "anger": {
        "angry", "mad", "furious", "annoyed", "frustrated", "irritated", "hate",
        "resentful", "offended", "bitter", "rage", "stupid", "awful", "fed up",
        "sick", "unfair", "argument", "snapped", "tense", "hostile",
    },
    "fear": {
        "afraid", "fear", "scared", "worried", "anxious", "anxiety", "panic",
        "stressed", "stress", "nervous", "uneasy", "overwhelmed", "terrified",
        "uncertain", "pressure", "dread", "shaking", "paranoid", "unsafe", "risk",
    },
    "love": {
        "love", "loved", "cared", "caring", "support", "supported", "warm",
        "close", "connected", "affection", "kind", "safe", "trust", "romantic",
        "gentle", "friendship", "family", "belonging", "cherish", "compassion",
    },
    "surprise": {
        "surprised", "shocked", "amazed", "unexpected", "suddenly", "wow",
        "astonished", "confused", "strange", "unbelievable", "startled",
        "curious", "caught", "abrupt", "different",
    },
    "neutral": {
        "okay", "normal", "average", "fine", "stable", "usual", "regular",
        "ordinary", "neutral", "balanced", "steady",
    },
}

PHRASE_HINTS = {
    "joy": ["feeling better", "looking forward", "a good day", "at peace"],
    "sadness": ["feel alone", "feel empty", "want to cry", "giving up"],
    "anger": ["so mad", "fed up", "lost my temper", "pissed off"],
    "fear": ["panic attack", "under pressure", "too much stress", "can not cope"],
    "love": ["feel supported", "feel loved", "close to", "care about"],
    "surprise": ["did not expect", "out of nowhere", "came suddenly"],
}

NEGATORS = {"not", "never", "hardly", "barely", "no", "without"}
INTENSIFIERS = {"very", "really", "extremely", "so", "quite", "too", "deeply"}

RISK_PATTERNS = [
    "i want to disappear",
    "i do not want to live",
    "i don't want to live",
    "hurt myself",
    "end my life",
    "kill myself",
    "self harm",
    "suicide",
]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?;\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    exps = {label: math.exp(value) for label, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {label: exps[label] / total for label in scores}


def compute_ehi(probabilities: Dict[str, float]) -> float:
    positive = probabilities.get("joy", 0.0) + probabilities.get("love", 0.0)
    negative = (
        probabilities.get("sadness", 0.0)
        + probabilities.get("anger", 0.0)
        + probabilities.get("fear", 0.0)
    )
    raw = 50 + (positive - negative) * 50
    return round(max(0.0, min(100.0, raw)), 1)


def infer_polarity(probabilities: Dict[str, float]) -> tuple[str, float]:
    positive = probabilities.get("joy", 0.0) + probabilities.get("love", 0.0)
    negative = (
        probabilities.get("sadness", 0.0)
        + probabilities.get("anger", 0.0)
        + probabilities.get("fear", 0.0)
    )
    label = "POSITIVE" if positive >= negative else "NEGATIVE"
    score = round(max(positive, negative), 4)
    return label, score


def make_support_message(top_label: str, risk_level: str) -> str:
    if risk_level == "high":
        return (
            "Your entry contains language associated with possible self-harm or severe distress. "
            "This tool is not a crisis service or diagnosis system. Please contact a trusted person, "
            "a local mental health professional, or emergency services now if you may be in immediate danger."
        )
    messages = {
        "joy": "You seem to be carrying some positive energy today. Try noting what helped.",
        "sadness": "Your text suggests sadness or emotional fatigue. Gentle rest and support may help.",
        "anger": "There may be frustration or tension here. A pause, walk, or breathing exercise could help.",
        "fear": "Your entry suggests stress or anxiety. Breaking tasks into smaller steps may reduce pressure.",
        "love": "There is a sense of care or connection in your writing. That support can be a real strength.",
        "surprise": "Something unexpected seems present. Writing one more sentence may help clarify the feeling.",
        "neutral": "Your emotions look relatively balanced right now. Tracking over time may reveal subtle shifts.",
    }
    return messages.get(top_label, messages["neutral"])


@dataclass
class SentenceAnalysis:
    text: str
    top_label: str
    top_score: float
    probs: List[Dict[str, float]]


class LexiconEmotionAnalyzer:
    label_to_id = {label: index for index, label in enumerate(EMOTIONS)}
    model_name = "offline-lexicon-analyzer"
    model_type = "heuristic"

    def _score_text(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        tokens = tokenize(text)
        counts = Counter(tokens)
        scores = {label: 0.25 for label in EMOTIONS}

        for label, words in EMOTION_LEXICON.items():
            for word in words:
                if " " in word:
                    continue
                base = counts.get(word, 0)
                if base:
                    scores[label] += base * 1.4

        for label, phrases in PHRASE_HINTS.items():
            for phrase in phrases:
                if phrase in text_lower:
                    scores[label] += 1.8

        for i, token in enumerate(tokens):
            if token not in NEGATORS:
                continue
            next_tokens = tokens[i + 1 : i + 4]
            for label, words in EMOTION_LEXICON.items():
                for candidate in next_tokens:
                    if candidate in words:
                        scores[label] -= 0.9
                        if label in {"joy", "love", "neutral"}:
                            scores["sadness"] += 0.7
                        else:
                            scores["neutral"] += 0.4

        for i, token in enumerate(tokens[:-1]):
            if token not in INTENSIFIERS:
                continue
            nxt = tokens[i + 1]
            for label, words in EMOTION_LEXICON.items():
                if nxt in words:
                    scores[label] += 0.8

        if len(tokens) <= 2:
            scores["neutral"] += 0.6

        if "but" in tokens or "however" in tokens:
            scores["surprise"] += 0.4

        return scores

    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        probabilities = softmax(self._score_text(sentence))
        probs = [
            {"label": label, "score": round(probabilities[label], 4)}
            for label in sorted(EMOTIONS, key=lambda label: probabilities[label], reverse=True)
        ]
        top = probs[0]
        return SentenceAnalysis(
            text=sentence,
            top_label=top["label"],
            top_score=top["score"],
            probs=probs,
        )

    def analyze(self, text: str) -> Dict[str, object]:
        sentences = split_sentences(text) or [text.strip()]
        sentence_results = [self.analyze_sentence(sentence) for sentence in sentences]

        aggregate = {label: 0.0 for label in EMOTIONS}
        for result in sentence_results:
            for item in result.probs:
                aggregate[item["label"]] += item["score"]

        count = float(len(sentence_results)) or 1.0
        aggregate = {label: round(value / count, 4) for label, value in aggregate.items()}
        ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
        label_name, score = ranked[0]
        polarity_label, polarity_score = infer_polarity(aggregate)
        risk_level = "high" if any(pattern in text.lower() for pattern in RISK_PATTERNS) else "normal"

        keywords = []
        tokens = tokenize(text)
        for label, words in EMOTION_LEXICON.items():
            for word in tokens:
                if word in words and word not in keywords:
                    keywords.append(word)
                if len(keywords) == 5:
                    break
            if len(keywords) == 5:
                break

        probs = [{"label": label, "score": aggregate[label]} for label, _ in ranked]
        active_labels = [
            {"label": item["label"], "score": item["score"]}
            for item in probs
            if item["label"] != "neutral" and item["score"] >= DEFAULT_MULTI_LABEL_THRESHOLD
        ]
        if not active_labels:
            active_labels = [{"label": label_name, "score": score}]
        return {
            "label_id": self.label_to_id[label_name],
            "label_name": label_name,
            "score": round(score, 4),
            "probs": probs,
            "sentences": [
                {
                    "text": result.text,
                    "top_label": result.top_label,
                    "top_score": result.top_score,
                    "probs": result.probs,
                }
                for result in sentence_results
            ],
            "polarity_label": polarity_label,
            "polarity_score": polarity_score,
            "ehi": compute_ehi(aggregate),
            "risk_level": risk_level,
            "support_message": make_support_message(label_name, risk_level),
            "keywords": keywords,
            "active_labels": active_labels,
            "multi_label_threshold": DEFAULT_MULTI_LABEL_THRESHOLD,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "privacy_mode": "local-only",
            "clinical_disclaimer": (
                "This prototype supports reflection and is not a medical diagnosis, treatment, "
                "or crisis intervention system."
            ),
        }


class TransformerEmotionAnalyzer:
    label_to_id = {label: index for index, label in enumerate(EMOTIONS)}
    model_type = "transformer"

    def __init__(self, model_dir: str | Path, threshold: float = DEFAULT_MULTI_LABEL_THRESHOLD):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is not installed. Install requirements-transformer.txt or use the lexicon fallback."
            ) from exc

        self.model_dir = str(model_dir)
        self.threshold = threshold
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True)
        self.classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            function_to_apply="sigmoid",
        )
        self.model_name = f"local-transformer:{Path(self.model_dir).name}"

    def _probabilities(self, text: str) -> Dict[str, float]:
        raw = self.classifier(text[:3000])[0]
        probabilities = {label: 0.0 for label in EMOTIONS}
        for item in raw:
            label = str(item["label"]).lower()
            if label.startswith("label_"):
                label = EMOTIONS[int(label.split("_", 1)[1])]
            if label in probabilities:
                probabilities[label] = float(item["score"])
        total = sum(probabilities.values())
        if total > 1.05:
            probabilities = {label: value / total for label, value in probabilities.items()}
        if probabilities.get("neutral", 0.0) == 0.0:
            used = sum(value for label, value in probabilities.items() if label != "neutral")
            probabilities["neutral"] = max(0.0, 1.0 - min(1.0, used))
        return probabilities

    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        probabilities = self._probabilities(sentence)
        probs = [
            {"label": label, "score": round(probabilities[label], 4)}
            for label in sorted(EMOTIONS, key=lambda label: probabilities[label], reverse=True)
        ]
        top = probs[0]
        return SentenceAnalysis(sentence, top["label"], top["score"], probs)

    def analyze(self, text: str) -> Dict[str, object]:
        sentences = split_sentences(text) or [text.strip()]
        sentence_results = [self.analyze_sentence(sentence) for sentence in sentences]
        probabilities = self._probabilities(text)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        label_name, score = ranked[0]
        probs = [{"label": label, "score": round(probabilities[label], 4)} for label, _ in ranked]
        active_labels = [
            {"label": item["label"], "score": item["score"]}
            for item in probs
            if item["label"] != "neutral" and item["score"] >= self.threshold
        ] or [{"label": label_name, "score": round(score, 4)}]
        polarity_label, polarity_score = infer_polarity(probabilities)
        risk_level = "high" if any(pattern in text.lower() for pattern in RISK_PATTERNS) else "normal"
        return {
            "label_id": self.label_to_id.get(label_name, 0),
            "label_name": label_name,
            "score": round(score, 4),
            "probs": probs,
            "sentences": [
                {
                    "text": result.text,
                    "top_label": result.top_label,
                    "top_score": result.top_score,
                    "probs": result.probs,
                }
                for result in sentence_results
            ],
            "polarity_label": polarity_label,
            "polarity_score": polarity_score,
            "ehi": compute_ehi(probabilities),
            "risk_level": risk_level,
            "support_message": make_support_message(label_name, risk_level),
            "keywords": [],
            "active_labels": active_labels,
            "multi_label_threshold": self.threshold,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "privacy_mode": "local-only",
            "clinical_disclaimer": (
                "This prototype supports reflection and is not a medical diagnosis, treatment, "
                "or crisis intervention system."
            ),
        }


def create_analyzer() -> LexiconEmotionAnalyzer:
    model_dir = os.environ.get("SENTIMENT_MODEL_DIR", "").strip()
    if model_dir:
        try:
            return TransformerEmotionAnalyzer(model_dir)  # type: ignore[return-value]
        except RuntimeError:
            pass
    return LexiconEmotionAnalyzer()
