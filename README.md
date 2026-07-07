# 🧠 DocMind

DocMind is an AI-powered document Q&A and study assistant. Upload PDF, TXT,
or DOCX files, ask questions grounded strictly in their content, and see
exactly which passages backed each answer. A **Quiz Me** mode turns the same
documents into self-check multiple-choice / short-answer quizzes with
instant, source-grounded feedback.

Built with **Streamlit**, **Google Gemini** (chat + embeddings), and
**ChromaDB** as a local, file-based vector store — no external services or
hosted databases required.

![DocMind screenshot placeholder](docs/screenshot.png)
*(Replace with a real screenshot once deployed — Q&A view with an answer and its expanded source snippets.)*

---

## Features

- 📁 Multi-file upload: PDF, TXT, DOCX
- ✂️ Automatic chunking (~500 tokens, ~50 token overlap) + embedding on upload
- 💬 Chat interface with multi-turn memory (within a session)
- 🔍 Every answer shows the exact source chunks + filenames it was grounded in
- 🚫 Refuses to answer from outside knowledge — says so explicitly when the docs don't contain the answer
- 📝 Quiz Me Mode: 5 auto-generated questions per round, graded instantly with explanations
- 📊 Live stats: documents indexed, chunks indexed, estimated tokens used
- 🗑️ One-click reset, isolated per browser session (safe for multiple concurrent users)

---

## Setup (local development)

**Requirements:** Python 3.11+ and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
# 1. Clone the repo
git clone <your-repo-url> docmind
cd docmind

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=your-real-key

# 4. Run it
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a document, click
**Process documents**, and start asking questions.

Your API key is read from `.env` locally (via `python-dotenv`) and from
`st.secrets` when deployed — it is never hardcoded and `.env` is
git-ignored.

---

## Deployment to Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick your repo/branch, and set **Main file path** to `app.py`.
4. Before deploying (or after, under **⋮ → Settings → Secrets**), add your secret in TOML format:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   ```
5. Click **Deploy**. No other configuration is needed — `requirements.txt`
   and the `.streamlit/config.toml` theme are picked up automatically.

**Note on ChromaDB + SQLite:** Streamlit Community Cloud's base image ships
an older system SQLite than ChromaDB requires. `requirements.txt` includes
`pysqlite3-binary` for Linux, and `app.py` swaps it in for the stdlib
`sqlite3` module at startup (a few lines at the very top of the file). This
is a no-op on Windows/macOS local dev, where the system SQLite is already
new enough.

---

## Architecture: the RAG pipeline

```
Upload → Extract → Chunk → Embed → Store (ChromaDB)
                                        ↑
Question → Embed → Vector search ──────┘
                       ↓
              Top-k relevant chunks
                       ↓
     Prompt = system instruction + chunks + question + history
                       ↓
                  Gemini (chat model)
                       ↓
         Answer, shown with its source chunks
```

**Ingestion** (`ingestion.py`): Each uploaded file is parsed by format
(`pypdf` for PDF, `python-docx` for DOCX, plain decode for TXT), then split
into overlapping chunks. Chunk size and overlap are specified in *tokens*,
approximated as ~4 characters/token — a lightweight heuristic that avoids
pulling in a full tokenizer dependency. Overlap matters because a fact
sitting at a chunk boundary could otherwise be split across two chunks and
retrieved poorly by either. Each chunk is embedded with Gemini's
`gemini-embedding-001` model (task type `RETRIEVAL_DOCUMENT`) and stored in
a ChromaDB collection alongside its source filename and chunk index.

**Session isolation:** ChromaDB's `PersistentClient` writes to one shared
on-disk store, but each browser session gets its own **collection** (named
by a per-session UUID). That means multiple people using the same deployed
app never see each other's documents, and one user's "reset" only deletes
their own collection.

**Retrieval** (`retrieval.py`): The user's question is embedded with the
same model, using task type `RETRIEVAL_QUERY` — Gemini optimizes the
embedding differently depending on whether it's indexing a document or
answering a query, which measurably improves match quality. ChromaDB then
returns the top-k (default 5) chunks by cosine similarity.

**Generation** (`llm.py`): The retrieved chunks, the current question, and
prior conversation turns are assembled into a Gemini `generate_content`
call. A system instruction constrains the model to answer **only** from the
supplied context and to explicitly say "I don't know based on the provided
documents" when the context doesn't contain the answer — this is what
keeps the assistant grounded instead of hallucinating from the model's
general knowledge. The UI renders the exact source chunks under each
answer so the user can verify the grounding themselves.

**Quiz Me Mode** (`quiz.py`): A random sample of indexed chunks is sent to
Gemini with a request for structured JSON output (`response_schema` bound
to Pydantic models), rather than free-form text that would need brittle
regex parsing. The API guarantees the response matches the schema, so
`quiz.py` gets back typed `QuizQuestion` objects directly — each carrying
its own question, options, correct answer, explanation, and source
snippet, so grading and citation both trace back to real document text.

---

## Project structure

```
docmind/
├── app.py          # Streamlit UI: sidebar, chat, quiz, session state
├── config.py        # Shared constants + API key resolution
├── ingestion.py      # Text extraction, chunking, embedding, Chroma storage
├── retrieval.py       # Query embedding + top-k similarity search
├── llm.py            # Gemini chat calls + grounded prompt construction
├── quiz.py            # Structured quiz generation + grading
├── requirements.txt
├── .env.example
└── .streamlit/config.toml
```

## Error handling

- **Unsupported file types** are rejected per-file with a clear sidebar warning; the rest of the batch still processes.
- **Empty/unreadable documents** (e.g. scanned image-only PDFs with no extractable text) are skipped with an explanation.
- **API failures / rate limits** from the Gemini API are retried with linear backoff, then surfaced as a readable error instead of crashing the app.
