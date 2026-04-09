# RAG Application

A simple **Retrieval-Augmented Generation (RAG)** app built with **LangChain**, **FastAPI**, and plain **HTML/JS**.

Upload a PDF or text document, then ask questions — the app retrieves relevant chunks and uses an LLM to answer.

---

## Architecture

```
┌──────────┐       ┌──────────────┐       ┌────────────┐
│  Browser  │──────▶│   FastAPI     │──────▶│  LangChain │
│  (HTML)   │◀──────│   Backend     │◀──────│  + FAISS   │
└──────────┘       └──────────────┘       └────────────┘
                          │                       │
                     Upload doc              OpenAI API
                     /query                 (embeddings + chat)
```

### How RAG Works (simplified)

1. **Upload** — The document is split into small overlapping chunks.
2. **Embed** — Each chunk is converted to a vector using OpenAI Embeddings.
3. **Store** — Vectors are stored in an in-memory FAISS index.
4. **Query** — The user's question is embedded, the top-k similar chunks are retrieved, and passed as context to the LLM which generates an answer.

---

## Setup

### 1. Clone / copy the project

```
RAG/
├── main.py              # FastAPI backend
├── static/
│   └── index.html       # Frontend
├── requirements.txt
├── .env.example
└── README.md
```

### 2. Create a virtual environment

```bash
cd RAG
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your real API key
```

### 5. Run the app

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## Usage

1. Click **Upload** and select a `.pdf` or `.txt` file.
2. Type a question in the text box and click **Ask**.
3. The answer and source chunks will appear below.

---

## Key Concepts for Students

| Concept | Where in code |
|---|---|
| Document loading | `load_document()` — uses LangChain's `PyPDFLoader` / `TextLoader` |
| Text chunking | `build_vector_store()` — `RecursiveCharacterTextSplitter` |
| Embeddings | `OpenAIEmbeddings()` converts text → vectors |
| Vector store | `FAISS.from_documents()` — similarity search index |
| Retrieval chain | `RetrievalQA.from_chain_type()` — retrieves context + generates answer |
| API endpoint | FastAPI `@app.post("/query")` |

---

## Notes

- This uses **in-memory** FAISS — data is lost on restart.
- Uses `gpt-3.5-turbo` by default. Change the model in `get_qa_chain()`.
- For production, add authentication, persistent storage, and rate limiting.
