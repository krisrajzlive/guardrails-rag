"""Llama Guard 3 1B wrapper used as both the input gate and the output gate.

Llama Guard is a classifier, not a chat model: fed a conversation, it replies
with "safe" or "unsafe" plus the violated MLCommons taxonomy category (S1-S14).
The same model checks the user's prompt (input gate) and the assistant's
answer (output gate) -- only the conversation passed to it differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import config

# https://huggingface.co/meta-llama/Llama-Guard-3-1B taxonomy
CATEGORY_LABELS = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex Crimes",
    "S4": "Child Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}


@dataclass
class GuardVerdict:
    is_safe: bool
    categories: list[str]
    raw: str

    @property
    def category_names(self) -> list[str]:
        return [CATEGORY_LABELS.get(c, c) for c in self.categories]


class LlamaGuard:
    """Lazy-loaded classifier so importing this module never triggers a model download."""

    def __init__(self, model_id: str = config.GUARD_MODEL_ID):
        self.model_id = model_id
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=config.HF_TOKEN)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=config.HF_TOKEN,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def _run_chat(self, chat: list[dict]) -> str:
        self._load()
        encoded = self._tokenizer.apply_chat_template(chat, return_tensors="pt", return_dict=True)
        input_ids = encoded["input_ids"]
        output = self._model.generate(**encoded, max_new_tokens=20, pad_token_id=0, do_sample=False)
        generated = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _parse(raw: str) -> GuardVerdict:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if not lines or lines[0].lower() == "safe":
            return GuardVerdict(is_safe=True, categories=[], raw=raw)
        categories = []
        if len(lines) > 1:
            categories = [c.strip() for c in lines[1].split(",") if c.strip()]
        return GuardVerdict(is_safe=False, categories=categories, raw=raw)

    @staticmethod
    def _msg(role: str, text: str) -> dict:
        # The model's chat template expects multimodal-style content parts
        # ([{"type": "text", "text": ...}]) -- a plain string silently renders
        # an empty conversation block instead of raising.
        return {"role": role, "content": [{"type": "text", "text": text}]}

    def check_input(self, user_text: str) -> GuardVerdict:
        return self._parse(self._run_chat([self._msg("user", user_text)]))

    def check_output(self, user_text: str, assistant_text: str) -> GuardVerdict:
        chat = [self._msg("user", user_text), self._msg("assistant", assistant_text)]
        return self._parse(self._run_chat(chat))


@lru_cache(maxsize=1)
def get_guard() -> LlamaGuard:
    return LlamaGuard()
