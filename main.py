import os
import shutil
import logging
import json
from pathlib import Path
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --------------- setup ---------------
load_dotenv()

# Structured Logging (Upgrade 12)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Application")

# --------------- constants & state ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_PATH = "faiss_index"
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Global State
vector_store: Optional[FAISS] = None
query_cache: dict[str, dict] = {} # Key: question (normalized), Value: {answer, sources}

# Conversation Memory (Upgrade 6)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

@app.on_event("startup")
async def startup_event():
    """Load the FAISS index on startup if it exists."""
    global vector_store
    if Path(INDEX_PATH).exists():
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.load_local(
                INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing FAISS index from {INDEX_PATH}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")

# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    return loader.load()


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    logger.info(f"Building vector store from {len(documents)} documents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def get_qa_chain(store, k=3, score_threshold=0.0):
    """Create a ConversationalRetrievalChain from the vector store."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # Custom Prompt Template (Upgrade 1)
    template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
If the context is completely unrelated to the topic of the question (e.g. math logic context for a question about a car crash), ignore the context and say you don't know based on the provided documents.
If the answer is not in the context, just say that you don't know.
Keep the answer concise and use markdown formatting where appropriate.

Context:
{context}

Question: {question}
Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Configure Retriever (Upgrade 4 & 11)
    if score_threshold > 0:
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold}
        )
    else:
        retriever = store.as_retriever(search_kwargs={"k": k})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
        get_chat_history=lambda h : h
    )

# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("static/index.html").read_text()


class QueryRequest(BaseModel):
    question: str
    k: int = 3
    score_threshold: float = 0.0


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        )
    
    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name
    
    logger.info(f"Uploading file: {safe_name}")
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        documents = load_document(str(file_path))
        new_store = build_vector_store(documents)
        
        if vector_store is None:
            vector_store = new_store
        else:
            # Multi-document support (Upgrade 5)
            logger.info("Merging new documents into existing vector store")
            vector_store.merge_from(new_store)
        
        # Persist the FAISS index (Upgrade 2)
        vector_store.save_local(INDEX_PATH)
        logger.info(f"FAISS index saved to {INDEX_PATH}")
        
        return {
            "message": f"'{safe_name}' uploaded and indexed successfully.",
            "pages": len(documents)
        }
    except Exception as e:
        logger.exception("Error during file upload and indexing")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List uploaded files (Upgrade 10)."""
    files = []
    for f in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.name):
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
    return {"documents": files}


@app.delete("/history")
async def clear_history():
    """Clear conversation history (Upgrade 6)."""
    memory.clear()
    return {"message": "Chat history cleared."}


@app.post("/query")
async def query_document(req: QueryRequest):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a document first.")
    
    # Query Caching (Upgrade 9)
    normalized_q = req.question.strip().lower()
    if normalized_q in query_cache:
        logger.info(f"Cache hit for query: {req.question}")
        cached = query_cache[normalized_q]
        # For simplicity, we'll return cached results as a single stream
        async def stream_cached():
            yield cached["answer"]
            # We add a separator to help the frontend identify sources if needed
            yield "\n\nSOURCES_METADATA:" + json.dumps(cached["sources"])
        return StreamingResponse(stream_cached(), media_type="text/plain")

    logger.info(f"Processing query: {req.question} (k={req.k}, threshold={req.score_threshold})")
    
    async def stream_answer():
        chain = get_qa_chain(vector_store, k=req.k, score_threshold=req.score_threshold)
        full_answer = ""
        source_docs = []
        
        try:
            # We use astream to get tokens. ConversationalRetrievalChain yields chunks.
            async for chunk in chain.astream({"question": req.question}):
                if "answer" in chunk:
                    token = chunk["answer"]
                    full_answer += token
                    yield token
                if "source_documents" in chunk:
                    source_docs = chunk["source_documents"]
            
            # Format sources for metadata yield
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:300],
                    "metadata": doc.metadata
                })
            
            # Cache the result (Upgrade 9)
            query_cache[normalized_q] = {"answer": full_answer, "sources": sources}
            
            # Yield sources as a special block at the end (Upgrade 3)
            yield "\n\nSOURCES_METADATA:" + json.dumps(sources)
            
        except Exception as e:
            logger.exception("Error during streaming query")
            yield f"\nERROR: {str(e)}"

    return StreamingResponse(stream_answer(), media_type="text/plain")
