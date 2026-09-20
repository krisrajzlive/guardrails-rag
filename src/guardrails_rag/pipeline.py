"""End-to-end guarded RAG pipeline.

    user query
       │
       ▼
  [INPUT GATE]  (a) the guard engine classifies the raw user prompt
                (b) dedicated prompt-injection classifier ALWAYS runs too --
                    neither guard engine's taxonomy covers injection at all
       │  either flags it → refuse, never touches the retriever or the LLM
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
(local Llama-Guard-3-1B inference via transformers). The injection
classifier (prompt_injection.py) is hosted via the HF Inference API and runs
regardless of GUARD_ENGINE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import config, ingest, rag_chain, secret_scrubber
from .prompt_injection import get_classifier


# The guard engine is pluggable (see config.GUARD_ENGINE): "openai_moderation"
def get_guard():
    if config.GUARD_ENGINE == "llama_guard":
        from .llama_guard import get_guard as _get_guard
    else:
        from .openai_moderation import get_guard as _get_guard
    return _get_guard()


# The result of running the guarded RAG pipeline on a single query. Contains the final answer (or refusal) and the verdicts from each stage.
@dataclass
class PipelineResult:
    answer: str
    blocked: bool
    block_stage: str | None
    input_verdict: Any  # llama_guard.GuardVerdict or openai_moderation.GuardVerdict
    output_verdict: Any | None
    injection_verdict: Any  # prompt_injection.InjectionVerdict
    secrets_redacted: dict[str, int]


REFUSAL = "I can't help with that request."


# The main entry point for the guarded RAG pipeline. Takes a user query, runs it through the input gate, retrieves context and generates an answer, and finally runs the output gate. 
# Returns a PipelineResult containing the final answer (or refusal) and the verdicts from each stage.
def run(query: str, top_k: int = 4) -> PipelineResult:

    # The pipeline consists of three main stages: input gate, retrieval and generation, and output gate. 
    # Each stage has its own checks to ensure that the query and the generated answer are safe and do not contain any sensitive information.
    guard = get_guard()
    injection_classifier = get_classifier()

    input_verdict = guard.check_input(query)
    injection_verdict = injection_classifier.check(query)

    # If either the input gate or the injection classifier flags the query, we refuse to process it further. 
    # This is a defense-in-depth approach to prevent malicious or unsafe queries from reaching the retrieval and generation stages of the pipeline. 
    # The input gate checks for unsafe content based on the
    if not input_verdict.is_safe:
        return PipelineResult(
            answer=REFUSAL,
            blocked=True,
            block_stage="input_gate_guard",
            input_verdict=input_verdict,
            output_verdict=None,
            injection_verdict=injection_verdict,
            secrets_redacted={},
        )

    if injection_verdict.is_injection:
        return PipelineResult(
            answer=REFUSAL,
            blocked=True,
            block_stage="input_gate_injection",
            input_verdict=input_verdict,
            output_verdict=None,
            injection_verdict=injection_verdict,
            secrets_redacted={},
        )

    # If the query passes both the input gate and the injection classifier, we proceed to retrieve context from the knowledge base and generate an answer using the RAG chain.
    context_chunks = ingest.retrieve_context(query, top_k=top_k)
    raw_answer = rag_chain.generate_answer(query, context_chunks)

    # After generating the raw answer, we check it against the output gate to ensure that the generated content is safe and does not contain any sensitive information.
    output_verdict = guard.check_output(query, raw_answer)
    scrub_result = secret_scrubber.scrub(raw_answer)

    # If the output gate flags the generated answer as unsafe, we refuse to return it to the user.
    # This is another layer of defense to prevent the dissemination of unsafe or sensitive content. 
    if not output_verdict.is_safe:
        return PipelineResult(
            answer=REFUSAL,
            blocked=True,
            block_stage="output_gate_guard",
            input_verdict=input_verdict,
            output_verdict=output_verdict,
            injection_verdict=injection_verdict,
            secrets_redacted=scrub_result.hits,
        )

    return PipelineResult(
        answer=scrub_result.text,
        blocked=False,
        block_stage=None,
        input_verdict=input_verdict,
        output_verdict=output_verdict,
        injection_verdict=injection_verdict,
        secrets_redacted=scrub_result.hits,
    )
