"""Interactive demo: shows both gates firing (or not) for each query."""
from __future__ import annotations

from . import pipeline

BANNER = """
Guarded RAG over the vendor NDA / security assessment PDF.
Type a question, or 'quit'. Try the sample attacks in README.md.
"""


def _print_result(result: pipeline.PipelineResult) -> None:
    print(f"[INPUT GATE]  safe={result.input_verdict.is_safe} "
          f"categories={result.input_verdict.category_names or '-'}")
    if result.output_verdict is not None:
        print(f"[OUTPUT GATE] safe={result.output_verdict.is_safe} "
              f"categories={result.output_verdict.category_names or '-'}")
    if result.secrets_redacted:
        print(f"[SCRUBBER]    redacted={result.secrets_redacted}")
    print(f"[BLOCKED]     {result.blocked}" + (f" at {result.block_stage}" if result.blocked else ""))
    print("\n" + result.answer + "\n")


def main() -> None:
    print(BANNER)
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit"}:
            break
        result = pipeline.run(query)
        _print_result(result)


if __name__ == "__main__":
    main()
