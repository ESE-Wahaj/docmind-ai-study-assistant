"""
Gemini chat calls: prompt construction + grounded answer generation.

This is the "G" (generation) in RAG. We take the chunks retrieval.py found,
wrap them in a system instruction that constrains the model to only use
that context, and send the running conversation history so follow-up
questions ("what about the second point?") still make sense.
"""

import time

from google import genai
from google.genai import types

import config

SYSTEM_INSTRUCTION = """You are DocMind, a study assistant that answers questions using ONLY the \
context excerpts provided alongside each question.

Rules:
- Only use information present in the given context to answer.
- If the answer cannot be found in the context, reply exactly: "I don't know based on the provided documents."
- Never use outside knowledge, even if you are confident about the answer.
- Be concise and direct.
- Where useful, briefly quote or paraphrase the specific part of the context that supports your answer.
"""


class LLMAPIError(Exception):
    """Raised when the Gemini generation API fails after retries."""


def _format_context(context_chunks: list[dict]) -> str:
    if not context_chunks:
        return "(No relevant context was found in the uploaded documents.)"
    return "\n\n".join(
        f"[Source: {c['filename']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in context_chunks
    )


def build_contents(
    chat_history: list[dict], question: str, context_chunks: list[dict]
) -> list[types.Content]:
    """Turn session chat history + retrieved context into Gemini's `contents` format.

    Prior turns are replayed as plain user/model messages (no re-attached
    context) so the model has conversational continuity; only the *current*
    question is paired with freshly retrieved context, since that context
    is re-retrieved every turn based on the latest question.
    """
    contents = []
    for turn in chat_history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))

    current_turn_text = (
        f"Context from uploaded documents:\n{_format_context(context_chunks)}\n\n"
        f"Question: {question}"
    )
    contents.append(types.Content(role="user", parts=[types.Part(text=current_turn_text)]))
    return contents


def generate_answer(
    genai_client: genai.Client,
    question: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    max_retries: int = 3,
) -> str:
    """Call Gemini to answer `question`, grounded in `context_chunks`."""
    contents = build_contents(chat_history, question, context_chunks)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model=config.CHAT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                ),
            )
            return response.text
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))

    raise LLMAPIError(
        f"Gemini generation failed after {max_retries} attempts: {last_error}"
    ) from last_error
