"""Hosted guard engine: OpenAI's Moderation API, used as both gates.

No local model weights, no download, no GPU/CPU inference -- every check is a
network call to OpenAI using the same API key already used for generation.
Categories are OpenAI's own taxonomy (harassment, hate, self-harm, sexual,
violence, ...), not the Llama Guard S1-S14 scheme.

The Moderation API takes a single block of text, not a user/assistant
conversation, so the output gate moderates the generated answer itself --
that's the text that would actually reach the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from . import config


@dataclass
class GuardVerdict:
    is_safe: bool
    categories: list[str]
    raw: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def category_names(self) -> list[str]:
        return self.categories


class OpenAIModerationGuard:
    def __init__(self, model: str = config.MODERATION_MODEL):
        self.model = model
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        from openai import OpenAI

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def _check(self, text: str) -> GuardVerdict:
        self._load()
        response = self._client.moderations.create(model=self.model, input=text)
        result = response.results[0]
        categories = result.categories.model_dump()
        scores = result.category_scores.model_dump()
        flagged = [name for name, hit in categories.items() if hit]
        return GuardVerdict(
            is_safe=not result.flagged,
            categories=flagged,
            raw=result.model_dump_json(),
            scores={name: scores[name] for name in flagged},
        )

    def check_input(self, user_text: str) -> GuardVerdict:
        return self._check(user_text)

    def check_output(self, user_text: str, assistant_text: str) -> GuardVerdict:
        return self._check(assistant_text)


@lru_cache(maxsize=1)
def get_guard() -> OpenAIModerationGuard:
    return OpenAIModerationGuard()
