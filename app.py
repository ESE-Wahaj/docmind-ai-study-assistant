"""
DocMind — an AI-powered document Q&A and study assistant.

Streamlit is used for both the frontend (widgets, chat UI) and the
"backend" (this script re-runs top-to-bottom on every interaction, and
long-lived state — chat history, the vector store handle, indexing
stats — is kept in st.session_state across those reruns).
"""

try:
    # ChromaDB requires SQLite >= 3.35. Streamlit Community Cloud's base
    # image ships an older system SQLite, so on Linux we swap in a modern
    # bundled build before chromadb (imported later, via ingestion.py) ever
    # touches sqlite3. Local dev on Windows/macOS already has a new enough
    # SQLite, and pysqlite3-binary isn't installed there, hence the guard.
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import uuid

import streamlit as st

import config
import ingestion
import llm
import quiz
import retrieval

st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")


def init_session_state():
    """Idempotent session defaults — safe to call on every rerun."""
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("collection_name", f"docmind_{st.session_state.session_id}")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("indexed_filenames", set())
    st.session_state.setdefault(
        "stats", {"files_indexed": 0, "chunks_indexed": 0, "estimated_tokens": 0}
    )
    st.session_state.setdefault("mode", "Q&A Mode")
    st.session_state.setdefault("quiz", None)
    st.session_state.setdefault("quiz_answers", {})
    st.session_state.setdefault("quiz_results", None)


def reset_session():
    """Wipe this session's vector data and chat/quiz state, start fresh.

    Each browser session gets its own ChromaDB *collection* (see
    collection_name above) inside one shared persistent store, so multiple
    people using the same deployed app never see each other's documents,
    and resetting one session can't wipe another user's data.
    """
    chroma_client = ingestion.get_chroma_client()
    try:
        chroma_client.delete_collection(name=st.session_state.collection_name)
    except Exception:
        pass  # collection may not exist yet (e.g. reset before any upload)

    for key in [
        "session_id",
        "collection_name",
        "messages",
        "indexed_filenames",
        "stats",
        "quiz",
        "quiz_answers",
        "quiz_results",
    ]:
        st.session_state.pop(key, None)
    init_session_state()


def render_sidebar(chroma_client) -> object:
    """Render sidebar controls and return this session's Chroma collection."""
    st.sidebar.title("🧠 DocMind")

    collection = ingestion.get_or_create_collection(chroma_client, st.session_state.collection_name)

    st.sidebar.subheader("📁 Upload documents")
    uploaded_files = st.sidebar.file_uploader(
        "PDF, TXT, or DOCX — multiple files supported",
        type=list(config.SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.sidebar.button("📥 Process documents", disabled=not uploaded_files, use_container_width=True):
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_filenames]
        if not new_files:
            st.sidebar.info("All uploaded files are already indexed.")
        else:
            with st.spinner(f"Embedding {len(new_files)} document(s)..."):
                try:
                    genai_client = ingestion.get_genai_client()
                    result = ingestion.process_uploaded_files(new_files, collection, genai_client)
                except ingestion.EmbeddingAPIError as exc:
                    st.sidebar.error(f"Failed to embed documents: {exc}")
                else:
                    st.session_state.stats["files_indexed"] += result["files_indexed"]
                    st.session_state.stats["chunks_indexed"] += result["chunks_indexed"]
                    st.session_state.stats["estimated_tokens"] += result["estimated_tokens"]
                    st.session_state.indexed_filenames.update(f.name for f in new_files)

                    for filename, reason in result["skipped"]:
                        st.sidebar.warning(f"Skipped **{filename}**: {reason}")
                    if result["files_indexed"] > 0:
                        st.sidebar.success(
                            f"Indexed {result['files_indexed']} file(s), "
                            f"{result['chunks_indexed']} chunks."
                        )

    st.sidebar.subheader("🧭 Mode")
    st.sidebar.radio(
        "Choose a mode",
        ["Q&A Mode", "Quiz Me Mode"],
        key="mode",
        label_visibility="collapsed",
    )

    st.sidebar.subheader("📊 Stats")
    stats = st.session_state.stats
    st.sidebar.markdown(
        f"- **Documents indexed:** {len(st.session_state.indexed_filenames)}\n"
        f"- **Chunks indexed:** {stats['chunks_indexed']}\n"
        f"- **Estimated tokens used:** {stats['estimated_tokens']:,}"
    )

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear documents / reset session", use_container_width=True):
        reset_session()
        st.rerun()

    return collection


def render_qa_mode(collection, genai_client):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['filename']}** — chunk {src['chunk_index']}")
                        st.markdown(f"> {src['text']}")

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    if collection.count() == 0:
        st.warning("Please upload and process at least one document first.")
        return

    # History sent to the model excludes the question we're about to answer —
    # generate_answer() attaches it separately, paired with fresh context.
    history_for_prompt = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        try:
            chunks = retrieval.retrieve_relevant_chunks(collection, genai_client, question)
            answer = llm.generate_answer(genai_client, question, chunks, history_for_prompt)
        except (ingestion.EmbeddingAPIError, llm.LLMAPIError) as exc:
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {exc}"})
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": chunks}
            )

    st.rerun()


def render_quiz_mode(collection, genai_client):
    if collection.count() == 0:
        st.info("Upload and process at least one document to generate a quiz.")
        return

    if st.button("🎲 Generate new quiz"):
        with st.spinner("Writing quiz questions from your documents..."):
            try:
                st.session_state.quiz = quiz.generate_quiz(genai_client, collection)
                st.session_state.quiz_answers = {}
                st.session_state.quiz_results = None
            except quiz.QuizGenerationError as exc:
                st.error(f"Couldn't generate a quiz: {exc}")

    if not st.session_state.quiz:
        st.info("Click **Generate new quiz** to get started.")
        return

    with st.form("quiz_form"):
        for idx, q in enumerate(st.session_state.quiz):
            st.markdown(f"**Q{idx + 1}. {q.question}**")
            if q.question_type == "multiple_choice":
                st.session_state.quiz_answers[idx] = st.radio(
                    "Choose one:", q.options, key=f"quiz_radio_{idx}", label_visibility="collapsed"
                )
            else:
                st.session_state.quiz_answers[idx] = st.text_input(
                    "Your answer:", key=f"quiz_text_{idx}", label_visibility="collapsed"
                )
            st.divider()
        submitted = st.form_submit_button("✅ Submit answers")

    if submitted:
        st.session_state.quiz_results = {
            idx: quiz.check_answer(q, st.session_state.quiz_answers.get(idx, ""))
            for idx, q in enumerate(st.session_state.quiz)
        }

    if st.session_state.quiz_results:
        results = st.session_state.quiz_results
        score = sum(results.values())
        st.subheader(f"Score: {score} / {len(st.session_state.quiz)}")

        for idx, q in enumerate(st.session_state.quiz):
            icon = "✅" if results.get(idx) else "❌"
            st.markdown(f"{icon} **Q{idx + 1}.** {q.question}")
            st.markdown(f"**Correct answer:** {q.correct_answer}")
            st.markdown(f"*Explanation:* {q.explanation}")
            st.markdown(f"> {q.source_snippet}")
            st.caption(f"Source: {q.source_filename}")
            st.divider()


def main():
    init_session_state()

    if not config.GEMINI_API_KEY:
        st.error(
            "**GEMINI_API_KEY is not set.** Add it to a local `.env` file "
            "(see `.env.example`) for local development, or to your app's "
            "Secrets on Streamlit Community Cloud."
        )
        st.stop()

    try:
        genai_client = ingestion.get_genai_client()
    except ingestion.EmbeddingAPIError as exc:
        st.error(str(exc))
        st.stop()

    chroma_client = ingestion.get_chroma_client()
    collection = render_sidebar(chroma_client)

    st.title("🧠 DocMind")
    st.caption("Upload your documents, then ask questions or quiz yourself — grounded in what you uploaded.")

    if st.session_state.mode == "Q&A Mode":
        render_qa_mode(collection, genai_client)
    else:
        render_quiz_mode(collection, genai_client)


if __name__ == "__main__":
    main()
