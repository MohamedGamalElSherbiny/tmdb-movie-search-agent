"""
LLM-based grounded answer generation using Gemini.

Hallucination mitigation:
  1. The prompt includes ONLY the retrieved movie documents -- never movies
     outside the retrieved set, never raw CSV rows.
  2. System prompt explicitly forbids adding facts not present in context.
  3. Numeric/statistical claims (ratings, revenue, counts) are never asked
     of this function -- those come from structured_search, deterministic.
  4. Empty or irrelevant context is handled explicitly: the model is told
     to say it couldn't find a match rather than answering from its own
     training knowledge of real movies.
"""
from __future__ import annotations

import os

from google import genai

SYSTEM_PROMPT = """You are a movie-discovery assistant. You answer ONLY using the \
movie context provided below. Rules:
- Never invent plot details, cast, ratings, or facts not present in the given context.
- If the context doesn't fully answer the question, say what's missing rather than guessing.
- When recommending movies, refer to them by the exact titles given in the context.
- Keep answers concise (3-6 sentences unless the user asked for a list).
- If no context is provided or it's irrelevant to the question, say you couldn't find a \
good match in the dataset rather than answering from general knowledge."""


def _format_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(no relevant movies retrieved)"
    blocks = []
    for i, m in enumerate(retrieved, 1):
        blocks.append(f"[{i}] (similarity={m['score']:.2f})\n{m['document']}")
    return "\n\n".join(blocks)


def generate_rag_answer(
    client: genai.Client,
    model_name: str,
    user_query: str,
    retrieved: list[dict],
) -> str:
    """retrieved: list of dicts with keys title, score, document (rag_document text)."""
    context_block = _format_context(retrieved)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Movie context (retrieved from the dataset):\n{context_block}\n\n"
        f"User question: {user_query}"
    )
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
    except Exception as e:  # noqa: BLE001 - surfaced to caller for graceful UI handling
        raise RuntimeError(f"LLM generation failed: {e}") from e

    return response.text