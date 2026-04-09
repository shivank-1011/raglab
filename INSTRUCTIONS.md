# RAG Application – Student Instructions

Complete each section in `main.py` by replacing `pass  # TODO` with your implementation.  
Refer to the hints, line descriptions, and documentation links below.

---

## 1. Imports

**What to do:** Import all the modules needed for the project.

| Import | Purpose |
|--------|---------|
| `os`, `shutil`, `pathlib.Path` | File system operations |
| `dotenv.load_dotenv` | Load environment variables from `.env` |
| `fastapi.FastAPI`, `File`, `UploadFile`, `HTTPException` | Web framework & error handling |
| `fastapi.responses.HTMLResponse` | Return HTML from a route |
| `fastapi.staticfiles.StaticFiles` | (Optional) Serve static files |
| `pydantic.BaseModel` | Request body validation |
| `langchain_community.document_loaders.PyPDFLoader` | Load PDF files |
| `langchain_community.document_loaders.TextLoader` | Load plain-text files |
| `langchain.text_splitter.RecursiveCharacterTextSplitter` | Split documents into chunks |
| `langchain_groq.ChatGroq` | Groq-hosted LLM wrapper |
| `langchain_community.embeddings.HuggingFaceEmbeddings` | Sentence-transformer embeddings |
| `langchain_community.vectorstores.FAISS` | FAISS vector store |
| `langchain.chains.RetrievalQA` | Retrieval-augmented QA chain |

**Documentation:**
- [FastAPI](https://fastapi.tiangolo.com/)
- [python-dotenv](https://saurabh-kumar.com/python-dotenv/)
- [LangChain](https://python.langchain.com/docs/introduction/)

---

## 2. Application & Global State Setup

**What to do:**
1. Call `load_dotenv()` to read the `.env` file.
2. Create a `FastAPI` instance with `title="RAG Application"`.
3. Create `UPLOAD_DIR = Path("uploads")` and call `.mkdir(exist_ok=True)`.
4. Declare `vector_store = None` (will hold the FAISS index after upload).

**Hints:**
- `Path.mkdir(exist_ok=True)` avoids errors if the directory already exists.

**Documentation:**
- [load_dotenv](https://saurabh-kumar.com/python-dotenv/)
- [FastAPI()](https://fastapi.tiangolo.com/reference/fastapi/)
- [pathlib.Path](https://docs.python.org/3/library/pathlib.html)

---

## 3. `load_document(file_path)`

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the uploaded file. Must end with `.pdf` or `.txt`. |

**Outputs:**
| Return | Type | Description |
|--------|------|-------------|
| documents | `list[Document]` | LangChain Document objects with `page_content` and `metadata`. |

**What to implement:**
1. Check if `file_path` ends with `".pdf"` → use `PyPDFLoader(file_path)`.
2. Else if it ends with `".txt"` → use `TextLoader(file_path, encoding="utf-8")`.
3. Otherwise raise `ValueError("Only .pdf and .txt files are supported.")`.
4. Return `loader.load()`.

**Hints:**
- `file_path.endswith(".pdf")` is the simplest extension check.
- `PyPDFLoader` produces one Document per page; `TextLoader` returns one Document.

**Documentation:**
- [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdf/)
- [TextLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.text.TextLoader.html)

---

## 4. `build_vector_store(documents)`

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `documents` | `list[Document]` | Documents returned by `load_document()`. |

**Outputs:**
| Return | Type | Description |
|--------|------|-------------|
| store | `FAISS` | A FAISS vector store usable as a retriever. |

**What to implement:**
1. Create `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
2. Split: `chunks = splitter.split_documents(documents)`.
3. Create `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`.
4. Build & return: `FAISS.from_documents(chunks, embeddings)`.

**Hints:**
- `chunk_size` = max characters per chunk.
- `chunk_overlap` = shared characters between adjacent chunks so context isn't lost at boundaries.
- `"all-MiniLM-L6-v2"` is a small, fast sentence-transformer model.

**Documentation:**
- [RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
- [HuggingFaceEmbeddings](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html)
- [FAISS.from_documents](https://python.langchain.com/docs/integrations/vectorstores/faiss/)

---

## 5. `get_qa_chain(store)`

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `store` | `FAISS` | Vector store from `build_vector_store()`. |

**Outputs:**
| Return | Type | Description |
|--------|------|-------------|
| chain | `RetrievalQA` | Chain callable via `chain.invoke({"query": "..."})`. Returns dict with `"result"` (str) and `"source_documents"` (list). |

**What to implement:**
1. `llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)`
2. Build chain with `RetrievalQA.from_chain_type`:
   - `llm` → the ChatGroq instance
   - `chain_type` → `"stuff"` (concatenates all retrieved docs into one prompt)
   - `retriever` → `store.as_retriever(search_kwargs={"k": 3})`
   - `return_source_documents` → `True`
3. Return the chain.

**Hints:**
- `temperature=0` makes the LLM deterministic.
- `"stuff"` is the simplest chain type – it stuffs all context into one prompt.
- `k=3` retrieves the top-3 most similar chunks.

**Documentation:**
- [ChatGroq](https://python.langchain.com/docs/integrations/chat/groq/)
- [RetrievalQA](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)
- [as_retriever](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore.as_retriever)

---

## 6. Home Route – `GET /`

**What to do:**
- Define `@app.get("/", response_class=HTMLResponse)` → `async def home():`
- Return `Path("static/index.html").read_text()`

**Documentation:**
- [FastAPI routes](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [HTMLResponse](https://fastapi.tiangolo.com/advanced/custom-response/#html-response)

---

## 7. Request Model – `QueryRequest`

**What to do:**
- Define a Pydantic model `QueryRequest` inheriting from `BaseModel`.
- Single field: `question: str`

**Documentation:**
- [Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)
- [FastAPI Request Body](https://fastapi.tiangolo.com/tutorial/body/)

---

## 8. Upload Route – `POST /upload`

**Inputs (HTTP):**
| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | `UploadFile` | Multipart form-data file upload. |

**Outputs (JSON):**
| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | Confirmation with filename. |
| `pages` | `int` | Number of documents the loader produced. |

**What to implement:**
1. Declare `global vector_store` so you can reassign it.
2. Validate `file.filename` is present (else `HTTPException(400)`).
3. Check extension via `Path(file.filename).suffix.lower()` — allow only `.pdf`, `.txt`.
4. Sanitize: `safe_name = Path(file.filename).name` (strips directory parts).
5. Save: open `UPLOAD_DIR / safe_name` in `"wb"` mode, use `shutil.copyfileobj(file.file, f)`.
6. Call `load_document(str(file_path))` and `build_vector_store(documents)`.
7. Wrap step 6 in try/except, raising `HTTPException(500, ...)` on failure.
8. Return `{"message": f"'{safe_name}' uploaded and indexed successfully.", "pages": len(documents)}`.

**Hints:**
- `Path(filename).name` prevents path-traversal attacks.
- `shutil.copyfileobj` streams the file efficiently.

**Documentation:**
- [UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/)
- [HTTPException](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [shutil.copyfileobj](https://docs.python.org/3/library/shutil.html#shutil.copyfileobj)

---

## 9. Query Route – `POST /query`

**Inputs (HTTP/JSON):**
| Parameter | Type | Description |
|-----------|------|-------------|
| `req` | `QueryRequest` | JSON body with `question` field. |

**Outputs (JSON):**
| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | LLM-generated answer. |
| `sources` | `list[dict]` | Each dict has `"content"` (first 300 chars) and `"metadata"`. |

**What to implement:**
1. If `vector_store is None` → raise `HTTPException(400, "No document uploaded yet. Please upload a document first.")`.
2. `chain = get_qa_chain(vector_store)`.
3. `result = chain.invoke({"query": req.question})`.
4. Build `sources` list by iterating `result.get("source_documents", [])`:
   - `"content"`: `doc.page_content[:300]`
   - `"metadata"`: `doc.metadata`
5. Return `{"answer": result["result"], "sources": sources}`.

**Hints:**
- `chain.invoke()` returns a dict with keys `"result"` and `"source_documents"`.

**Documentation:**
- [RetrievalQA.invoke](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)
- [FastAPI POST route](https://fastapi.tiangolo.com/tutorial/body/)
