"""Dedicated prompt-injection/jailbreak classifier, hosted via the HF Inference
API (no local download, no GPU/CPU inference).

Neither guard engine's taxonomy (Llama Guard's S1-S14, OpenAI's harassment/
hate/self-harm/sexual/violence) covers "ignore your instructions"-style
injection at all -- it's a structurally different risk from toxic content.
This classifier's only job is that one risk, and it separates cleanly:
>99% SAFE on benign RAG questions, 99.99% INJECTION on the attack prompt
tested during development (see README).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import config


@dataclass
class InjectionVerdict:
    is_injection: bool
    score: float
    raw_label: str


class PromptInjectionClassifier:
    def __init__(self, model: str = config.INJECTION_MODEL, threshold: float = config.INJECTION_THRESHOLD):
        self.model = model
        self.threshold = threshold
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        from huggingface_hub import InferenceClient

        self._client = InferenceClient(token=config.HF_TOKEN)

    def check(self, text: str) -> InjectionVerdict:
        self._load()
        results = self._client.text_classification(text, model=self.model)
        top = max(results, key=lambda r: r.score)
        is_injection = top.label.upper() == "INJECTION" and top.score >= self.threshold
        return InjectionVerdict(is_injection=is_injection, score=top.score, raw_label=top.label)


@lru_cache(maxsize=1)
def get_classifier() -> PromptInjectionClassifier:
    return PromptInjectionClassifier()
