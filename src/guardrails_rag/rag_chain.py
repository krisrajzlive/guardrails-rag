"""LangChain half of the pipeline: turns retrieved chunks + a question into an answer.

Retrieval lives in ingest.py (LlamaIndex); this module only does prompt
construction and LLM invocation, so swapping the generation model never touches
the indexing code.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

from . import config

SYSTEM_PROMPT = (
    "You are a vendor-onboarding assistant. Answer the question using ONLY the "
    "provided context from the vendor security assessment document. If the answer "
    "isn't in the context, say you don't know. Quote the context verbatim when asked "
    "for specific values."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@lru_cache(maxsize=1)
def _get_llm():
    if config.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.GEN_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)

    # No OpenAI key: fall back to a small local HF text-generation model so the
    # demo still runs end-to-end offline.
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    llm = _get_llm()
    chain = PROMPT | llm
    result = chain.invoke({"context": context, "question": question})
    return getattr(result, "content", result)
