"""Deterministic, regex-based output filter.

Llama Guard's taxonomy (S1-S14) has no category dedicated to "verbatim credential
leakage" -- a RAG system happily quoting an API key or SSH private key it found in
a retrieved chunk can slip past it as "safe". This module is a second, rule-based
output gate that runs regardless of the Llama Guard verdict, so a secret can never
reach the user just because the classifier missed it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS: dict[str, re.Pattern] = {
    "AWS_ACCESS_KEY": re.compile(r"AKIA[0-9A-Z]{16}"),
    "AWS_SECRET_KEY": re.compile(r"(?i)aws_secret_access_key['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{30,}"),
    "GENERIC_API_KEY": re.compile(r"(?i)\b(?:ak_live|sk_live|api[_-]?key|x-api-key)['\"]?\s*[:=]?\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    "SSH_PRIVATE_KEY": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    "PASSWORD_ASSIGNMENT": re.compile(r"(?i)\b(?:db_pass|password|passwd)['\"]?\s*[:=]\s*['\"]?\S{6,}"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


@dataclass
class ScrubResult:
    text: str
    hits: dict[str, int] = field(default_factory=dict)

    @property
    def had_secrets(self) -> bool:
        return bool(self.hits)


def scrub(text: str) -> ScrubResult:
    hits: dict[str, int] = {}
    for label, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            hits[label] = len(matches)
            text = pattern.sub(f"[REDACTED:{label}]", text)
    return ScrubResult(text=text, hits=hits)
