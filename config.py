"""
Central configuration for DocMind.

Keeping model names, chunking parameters, and the API key lookup in one
place means every other module (ingestion, retrieval, llm, quiz) imports
from here instead of hardcoding values — change a model name or chunk
size once and it applies everywhere.
"""

import os

import streamlit as st
from dotenv import load_dotenv

# Load variables from a local .env file if present. On Streamlit Community
# Cloud there is no .env file — secrets are injected via st.secrets instead.
load_dotenv()


def get_gemini_api_key() -> str | None:
    """Resolve the Gemini API key from Streamlit secrets first, then env vars.

    st.secrets is how Streamlit Community Cloud injects secrets at runtime;
    the environment variable / .env path is what we use for local dev. We
    never hardcode the key in source.
    """
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        # st.secrets raises if no secrets.toml exists at all (e.g. local dev
        # without Streamlit secrets configured) - fall back to env vars.
        pass
    return os.environ.get("GEMINI_API_KEY")


GEMINI_API_KEY = get_gemini_api_key()

# --- Gemini models ---------------------------------------------------------
# gemini-embedding-001 is Google's current general-availability embedding
# model. We truncate its native 3072-dim output to 768 dims (Matryoshka
# Representation Learning support) to keep the ChromaDB index smaller and
# queries faster, with a negligible quality tradeoff for this use case.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

# gemini-2.5-flash is a strong, low-cost, low-latency model well suited to
# grounded Q&A and quiz generation.
CHAT_MODEL = "gemini-2.5-flash"

# --- Chunking ----------------------------------------------------------------
# We approximate tokens as ~4 characters (a common rule-of-thumb for
# English text) rather than pulling in a full tokenizer dependency. It's
# not exact, but it's consistent and good enough to size chunks sensibly
# and to show a ballpark "tokens used" stat in the UI.
CHARS_PER_TOKEN = 4
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# --- Retrieval -----------------------------------------------------------
TOP_K_CHUNKS = 5

# --- Storage ---------------------------------------------------------------
CHROMA_PERSIST_DIR = "chroma_db"

# --- Quiz ------------------------------------------------------------------
QUIZ_NUM_QUESTIONS = 5

SUPPORTED_EXTENSIONS = {"pdf", "txt", "docx"}
