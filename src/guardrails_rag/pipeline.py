"""End-to-end guarded RAG pipeline.

    user query
       │
       ▼
  [INPUT GATE]  Llama Guard 3 1B classifies the raw user prompt
       │  unsafe → refuse, never touches the retriever or the LLM
       ▼ safe
  LlamaIndex retrieval  +  LangChain generation
       │
       ▼
  [OUTPUT GATE] (a) Llama Guard 3 1B classifies (prompt, answer)
                (b) regex secret_scrubber ALWAYS redacts credentials/PII,
                    independent of the Llama Guard verdict (defense in depth)
       │
       ▼
    final answer or refusal, with the verdicts surfaced to the caller
"""
from __future__ import annotations

from dataclasses import dataclass

from . import ingest, rag_chain, secret_scrubber
from .llama_guard import GuardVerdict, get_guard


@dataclass
class PipelineResult:
    answer: str
    blocked: bool
    block_stage: str | None
    input_verdict: GuardVerdict
    output_verdict: GuardVerdict | None
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
            block_stage="output_gate_llama_guard",
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
