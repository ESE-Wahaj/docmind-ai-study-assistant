"""
Retrieval: turn a user question into the most relevant document chunks.

This is the "R" in RAG. Given a question, we embed it with the same model
used at ingestion time (but a different task_type — see ingestion.py) and
ask ChromaDB for the nearest neighbor chunks by cosine similarity.
"""

from google import genai

import config
from ingestion import embed_texts


def retrieve_relevant_chunks(
    collection,
    genai_client: genai.Client,
    question: str,
    top_k: int = config.TOP_K_CHUNKS,
) -> list[dict]:
    """Embed the question and fetch the top_k most similar chunks.

    Returns a list of dicts: {"text", "filename", "chunk_index", "distance"},
    ordered from most to least relevant. Distance is cosine distance
    (lower = more similar), surfaced so the UI/LLM prompt can reason about
    confidence if needed.
    """
    if collection.count() == 0:
        return []

    [question_embedding] = embed_texts(
        genai_client, [question], task_type="RETRIEVAL_QUERY"
    )

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "filename": metadata["filename"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return chunks
