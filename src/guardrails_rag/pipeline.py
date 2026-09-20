"""End-to-end guarded RAG pipeline.

    user query
       │
       ▼
  [INPUT GATE]  the guard engine classifies the raw user prompt
       │  unsafe → refuse, never touches the retriever or the LLM
       ▼ safe
  LlamaIndex retrieval  +  LangChain generation
       │
       ▼
  [OUTPUT GATE] (a) the guard engine classifies the generated answer
                (b) regex secret_scrubber ALWAYS redacts credentials/PII,
                    independent of the guard verdict (defense in depth)
       │
       ▼
    final answer or refusal, with the verdicts surfaced to the caller

The guard engine is pluggable (see config.GUARD_ENGINE): "openai_moderation"
(default -- hosted, no local model download/inference) or "llama_guard"
(local Llama-Guard-3-1B inference via transformers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import config, ingest, rag_chain, secret_scrubber


def get_guard():
    if config.GUARD_ENGINE == "llama_guard":
        from .llama_guard import get_guard as _get_guard
    else:
        from .openai_moderation import get_guard as _get_guard
    return _get_guard()


@dataclass
class PipelineResult:
    answer: str
    blocked: bool
    block_stage: str | None
    input_verdict: Any  # llama_guard.GuardVerdict or openai_moderation.GuardVerdict
    output_verdict: Any | None
    secrets_redacted: dict[str, int]


REFUSAL = "I can't help with that request."


def run(query: str, top_k: int = 4) -> PipelineResult:
    guard = get_guard()

    input_verdict = guard.check_input(query)
    if not input_verdict.is_safe:
        return PipelineResult(
            answer=REFUSAL,
            blocked=True,
            block_stage="input_gate",
            input_verdict=input_verdict,
            output_verdict=None,
            secrets_redacted={},
        )

    context_chunks = ingest.retrieve_context(query, top_k=top_k)
    raw_answer = rag_chain.generate_answer(query, context_chunks)

    output_verdict = guard.check_output(query, raw_answer)
    scrub_result = secret_scrubber.scrub(raw_answer)

    if not output_verdict.is_safe:
        return PipelineResult(
            answer=REFUSAL,
            blocked=True,
            block_stage="output_gate_guard",
            input_verdict=input_verdict,
            output_verdict=output_verdict,
            secrets_redacted=scrub_result.hits,
        )

    return PipelineResult(
        answer=scrub_result.text,
        blocked=False,
        block_stage=None,
        input_verdict=input_verdict,
        output_verdict=output_verdict,
        secrets_redacted=scrub_result.hits,
    )
