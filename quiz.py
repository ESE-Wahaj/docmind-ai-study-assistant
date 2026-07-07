"""
Quiz Me Mode: generate self-check questions grounded in the uploaded docs.

Rather than asking Gemini for free-form text and regex-parsing it, we use
the Gemini API's structured output mode (response_schema) with Pydantic
models. The API guarantees the response matches the schema, so we get
back typed QuizQuestion objects directly with no brittle parsing.
"""

import random
import time
from typing import Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

import config

MAX_CONTEXT_CHUNKS = 30


class QuizGenerationError(Exception):
    """Raised when quiz generation fails after retries."""


class QuizQuestion(BaseModel):
    question: str
    question_type: Literal["multiple_choice", "short_answer"]
    options: Optional[list[str]] = None
    correct_answer: str
    explanation: str
    source_filename: str
    source_snippet: str


class QuizSet(BaseModel):
    questions: list[QuizQuestion]


def _sample_chunks(collection, max_chunks: int = MAX_CONTEXT_CHUNKS) -> list[dict]:
    """Pull a random sample of indexed chunks to ground the quiz in.

    Capping the sample keeps the generation prompt (and cost) bounded even
    for large document sets, while random sampling avoids always quizzing
    on just the first file uploaded.
    """
    if collection.count() == 0:
        return []

    result = collection.get(include=["documents", "metadatas"])
    documents = result["documents"]
    metadatas = result["metadatas"]

    indices = list(range(len(documents)))
    if len(indices) > max_chunks:
        indices = random.sample(indices, max_chunks)

    return [
        {
            "text": documents[i],
            "filename": metadatas[i]["filename"],
            "chunk_index": metadatas[i]["chunk_index"],
        }
        for i in indices
    ]


def generate_quiz(
    genai_client: genai.Client,
    collection,
    num_questions: int = config.QUIZ_NUM_QUESTIONS,
    max_retries: int = 3,
) -> list[QuizQuestion]:
    """Generate a quiz grounded in a random sample of indexed document chunks."""
    sampled_chunks = _sample_chunks(collection)
    if not sampled_chunks:
        raise QuizGenerationError("No documents have been indexed yet.")

    context_block = "\n\n".join(
        f"[Source: {c['filename']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in sampled_chunks
    )

    prompt = f"""Using ONLY the context excerpts below, write exactly {num_questions} quiz \
questions that test understanding of the material. Mix "multiple_choice" (exactly 4 \
options, exactly one correct) and "short_answer" question types.

For every question provide:
- question: the question text
- question_type: "multiple_choice" or "short_answer"
- options: a list of exactly 4 strings for multiple_choice, or null for short_answer
- correct_answer: the correct option text (for multiple_choice) or a concise correct answer (for short_answer)
- explanation: a short explanation grounded in the context, of why that answer is correct
- source_filename: the filename of the source excerpt the question is based on
- source_snippet: the exact excerpt (verbatim from the context) the question is based on

Do not invent facts that aren't in the context.

Context:
{context_block}
"""

    last_error = None
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model=config.CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QuizSet,
                    temperature=0.4,
                ),
            )
            quiz_set = response.parsed or QuizSet.model_validate_json(response.text)
            if not quiz_set.questions:
                raise ValueError("Model returned zero questions.")
            return quiz_set.questions
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))

    raise QuizGenerationError(
        f"Quiz generation failed after {max_retries} attempts: {last_error}"
    ) from last_error


def check_answer(question: QuizQuestion, user_answer: str) -> bool:
    """Grade a user's answer.

    Multiple-choice is an exact match against the chosen option. Short-answer
    grading is a lenient, non-LLM lexical containment check (case-insensitive
    substring match either direction) rather than a full semantic judgment —
    the correct answer and explanation are always shown alongside the result
    so the user can make the final call themselves.
    """
    user_normalized = user_answer.strip().lower()
    correct_normalized = question.correct_answer.strip().lower()

    if not user_normalized:
        return False

    if question.question_type == "multiple_choice":
        return user_normalized == correct_normalized

    return user_normalized in correct_normalized or correct_normalized in user_normalized
