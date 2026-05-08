from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


EMOTIONS = ["joy", "sadness", "anger", "fear", "love", "surprise", "neutral"]

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
            "Your entry suggests strong distress. Please consider reaching out to a "
            "trusted person or a local mental health professional right away."
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
            "model_name": "offline-lexicon-analyzer",
            "privacy_mode": "local-only",
        }


def create_analyzer() -> LexiconEmotionAnalyzer:
    return LexiconEmotionAnalyzer()
