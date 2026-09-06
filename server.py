"""FastAPI backend exposing the self-healing RAG graph as a streaming API.

Each LangGraph node is streamed to the client as a server-sent event the moment it
finishes, so the frontend can render the retrieve/generate/critique loop as it runs
rather than waiting for a final answer.
"""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import config
from rag.graph import build_graph
from rag.ingest import (
    SUPPORTED_SUFFIXES,
    DocumentError,
    add_document,
    clear_collection,
    collection_stats,
    get_embeddings,
    get_vector_store,
)
from rag.llm import validate_credentials
from rag.state import RAGState

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_runtime: Dict[str, Any] = {}


def _graph():
    """Compiled lazily and cached; invalidated whenever the corpus changes."""
    if "graph" not in _runtime:
        _runtime["graph"] = build_graph(get_vector_store())
    return _runtime["graph"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        validate_credentials()
    except Exception as exc:
        logging.warning(f"Credential validation note at startup: {exc}")
    yield
    _runtime.clear()


app = FastAPI(title="Self-Healing RAG", lifespan=lifespan)

# Allow CORS for local dev servers and cloud deployments (like Hugging Face Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    max_attempts: int = config.MAX_ATTEMPTS


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _serialize_chunks(docs) -> List[dict]:
    return [
        {"source": d.metadata.get("source", "unknown"), "content": d.page_content}
        for d in docs
    ]


def _event_stream(question: str, max_attempts: int) -> Iterator[str]:
    initial: RAGState = {
        "question": question,
        "current_query": question,
        "retrieved_docs": [],
        "last_retrieval_query": None,
        "answer": "",
        "critic_verdict": None,
        "attempts": 0,
        "max_attempts": max_attempts,
        "trace": [],
        "final_answer": None,
        "status": None,
    }

    yield _sse({"type": "start", "question": question, "maxAttempts": max_attempts})

    try:
        for update in _graph().stream(initial, stream_mode="updates"):
            for node, delta in update.items():
                if node == "retrieve":
                    yield _sse({
                        "type": "retrieve",
                        "query": delta["last_retrieval_query"],
                        "chunks": _serialize_chunks(delta["retrieved_docs"]),
                    })
                elif node == "generate":
                    yield _sse({"type": "generate", "answer": delta["answer"]})
                elif node == "critique":
                    verdict = delta["critic_verdict"]
                    yield _sse({
                        "type": "critique",
                        "attempt": delta["attempts"],
                        "verdict": verdict.verdict,
                        "reasoning": verdict.reasoning,
                        "reformulatedQuery": verdict.reformulated_query,
                    })
                elif node in ("finalize", "refuse"):
                    yield _sse({
                        "type": "final",
                        "status": delta["status"],
                        "answer": delta["final_answer"],
                    })
    except Exception as exc:  # surface failures in the UI instead of a dead stream
        yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    yield _sse({"type": "done"})


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    if not req.question.strip():
        raise HTTPException(400, "Question is empty.")
    if collection_stats()["count"] == 0:
        raise HTTPException(400, "No documents yet — upload one to build the corpus.")

    return StreamingResponse(
        _event_stream(req.question.strip(), req.max_attempts),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)) -> dict:
    results = []
    for f in files:
        name = f.filename or "unnamed"
        data = await f.read()

        if not name.lower().endswith(SUPPORTED_SUFFIXES):
            results.append({
                "filename": name,
                "ok": False,
                "error": f"Unsupported type. Accepted: {', '.join(SUPPORTED_SUFFIXES)}.",
            })
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            results.append({
                "filename": name,
                "ok": False,
                "error": f"Exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            })
            continue

        try:
            t0 = time.time()
            print(f"[Upload] Processing '{name}' ({len(data)} bytes)...", flush=True)
            # Run CPU-bound text extraction and embedding in a separate thread
            # so the FastAPI event loop is not blocked
            chunks = await asyncio.to_thread(add_document, data, name)
            elapsed = time.time() - t0
            print(f"[Upload] Added {chunks} chunks for '{name}' in {elapsed:.2f}s", flush=True)
            results.append({"filename": name, "ok": True, "chunks": chunks})
        except DocumentError as exc:
            print(f"[Upload] DocumentError for '{name}': {exc}", flush=True)
            results.append({"filename": name, "ok": False, "error": str(exc)})
        except Exception as exc:
            print(f"[Upload] Unexpected error for '{name}': {exc}", flush=True)
            results.append({"filename": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    # Force a rebuild so the next question sees the newly embedded chunks.
    _runtime.pop("graph", None)
    stats = await asyncio.to_thread(collection_stats)
    return {"results": results, "corpus": stats}


@app.delete("/api/corpus")
def clear_corpus() -> dict:
    clear_collection()
    _runtime.pop("graph", None)
    return {"ok": True, "corpus": collection_stats()}


@app.get("/api/corpus")
def corpus() -> dict:
    return collection_stats()


@app.get("/api/health")
def health() -> dict:
    return {
        "provider": config.LLM_PROVIDER,
        "generationModel": config.GENERATION_MODEL,
        "criticModel": config.CRITIC_MODEL,
        "embeddingModel": config.EMBEDDING_MODEL,
        "topK": config.TOP_K,
        "maxAttempts": config.MAX_ATTEMPTS,
        "accepts": list(SUPPORTED_SUFFIXES),
        "corpus": collection_stats(),
    }


# Serve built React frontend if dist exists (e.g. in Docker or production)
_dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(_dist_dir):
    app.mount("/", StaticFiles(directory=_dist_dir, html=True), name="frontend")
