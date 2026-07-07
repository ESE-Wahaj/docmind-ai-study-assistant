"""
Ingestion pipeline: turn uploaded files into searchable vectors.

Pipeline for each uploaded file:
    1. Extract raw text (format-specific: PDF / TXT / DOCX).
    2. Split the text into overlapping chunks sized for retrieval.
    3. Embed each chunk with Gemini's embedding model.
    4. Store the chunk text + embedding + metadata in a ChromaDB collection.

Chunking with overlap matters because a single fact can straddle a chunk
boundary; without overlap, the sentence containing the answer might get
cut in half and neither half would score well against the query.
"""

import io
import time
import uuid

import chromadb
import streamlit as st
from docx import Document as DocxDocument
from google import genai
from google.genai import types
from pypdf import PdfReader

import config


class UnsupportedFileTypeError(Exception):
    """Raised when an uploaded file's extension isn't one we know how to parse."""


class EmptyDocumentError(Exception):
    """Raised when a file parses successfully but contains no extractable text."""


class EmbeddingAPIError(Exception):
    """Raised when the Gemini embedding API fails after retries."""


# --- Text extraction ---------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Dispatch to the right parser based on file extension."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in config.SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"'.{extension}' is not supported. Please upload PDF, TXT, or DOCX files."
        )

    if extension == "pdf":
        text = extract_text_from_pdf(file_bytes)
    elif extension == "docx":
        text = extract_text_from_docx(file_bytes)
    else:  # txt
        text = extract_text_from_txt(file_bytes)

    if not text.strip():
        raise EmptyDocumentError(
            f"No extractable text found in '{filename}' "
            "(it may be a scanned/image-only document)."
        )
    return text


# --- Chunking ----------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate using the ~4-chars-per-token heuristic (see config.py)."""
    return max(1, len(text) // config.CHARS_PER_TOKEN)


def chunk_text(
    text: str,
    chunk_size_tokens: int = config.CHUNK_SIZE_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping, word-boundary-safe chunks.

    We walk the text word by word (never splitting mid-word) and cut a new
    chunk once we cross the target character budget for chunk_size_tokens.
    The last few words of each chunk are carried over into the next one so
    consecutive chunks overlap by roughly overlap_tokens worth of context.
    """
    chunk_size_chars = chunk_size_tokens * config.CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * config.CHARS_PER_TOKEN

    words = text.split()
    if not words:
        return []

    chunks = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        current_words.append(word)
        current_len += len(word) + 1  # +1 for the joining space

        if current_len >= chunk_size_chars:
            chunks.append(" ".join(current_words))

            # Carry the trailing ~overlap_chars worth of words into the next chunk.
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_words):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap_chars:
                    break

            current_words = overlap_words
            current_len = sum(len(w) + 1 for w in current_words)

    # Final partial chunk, if any words are left over (skip if it would be
    # an exact duplicate of the chunk we just emitted).
    if current_words:
        remainder = " ".join(current_words)
        if not chunks or remainder != chunks[-1]:
            chunks.append(remainder)

    return chunks


# --- Embeddings ----------------------------------------------------------

@st.cache_resource
def get_genai_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise EmbeddingAPIError(
            "GEMINI_API_KEY is not set. Add it to a local .env file or to "
            "Streamlit secrets before uploading documents."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def embed_texts(
    client: genai.Client,
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 50,
    max_retries: int = 3,
) -> list[list[float]]:
    """Embed a list of strings with Gemini, batching to stay under API limits.

    task_type differentiates how Gemini optimizes the embedding: documents
    being indexed use RETRIEVAL_DOCUMENT, while user questions at query time
    use RETRIEVAL_QUERY (see retrieval.py). Using the matching task_type for
    each side measurably improves retrieval quality.
    """
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        last_error = None

        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=config.EMBEDDING_DIMENSIONS,
                    ),
                )
                all_embeddings.extend(e.values for e in response.embeddings)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # simple linear backoff

        if last_error is not None:
            raise EmbeddingAPIError(
                f"Gemini embedding call failed after {max_retries} attempts: {last_error}"
            ) from last_error

    return all_embeddings


# --- ChromaDB storage ---------------------------------------------------------

@st.cache_resource
def get_chroma_client() -> chromadb.ClientAPI:
    """A single persistent Chroma client shared across reruns and sessions.

    Session isolation comes from using a per-session *collection name*
    (see app.py), not from separate clients — many collections can safely
    live in one persistent store.
    """
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_or_create_collection(client: chromadb.ClientAPI, collection_name: str):
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def process_uploaded_files(
    uploaded_files,
    collection,
    genai_client: genai.Client,
) -> dict:
    """Run the full ingestion pipeline for a batch of uploaded files.

    Returns a stats dict: {"files_indexed", "chunks_indexed",
    "estimated_tokens", "skipped": [(filename, reason), ...]}.
    Errors on individual files (bad format, empty content) are collected
    as "skipped" rather than aborting the whole batch.
    """
    files_indexed = 0
    chunks_indexed = 0
    estimated_tokens = 0
    skipped: list[tuple[str, str]] = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        try:
            text = extract_text(filename, uploaded_file.getvalue())
        except (UnsupportedFileTypeError, EmptyDocumentError) as exc:
            skipped.append((filename, str(exc)))
            continue

        chunks = chunk_text(text)
        if not chunks:
            skipped.append((filename, "produced no chunks after processing."))
            continue

        embeddings = embed_texts(genai_client, chunks, task_type="RETRIEVAL_DOCUMENT")

        ids = [f"{filename}::{i}::{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        files_indexed += 1
        chunks_indexed += len(chunks)
        estimated_tokens += sum(estimate_tokens(c) for c in chunks)

    return {
        "files_indexed": files_indexed,
        "chunks_indexed": chunks_indexed,
        "estimated_tokens": estimated_tokens,
        "skipped": skipped,
    }
