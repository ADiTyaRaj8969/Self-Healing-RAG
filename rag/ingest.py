import io
import math
import threading
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config

# One corpus, populated entirely by the user's own uploads.
COLLECTION = "docs"

TEXT_SUFFIXES = (".txt", ".md", ".markdown")
SUPPORTED_SUFFIXES = (".pdf",) + TEXT_SUFFIXES

_embeddings: "OnnxMiniLMEmbeddings | None" = None
_vector_store: Chroma | None = None

# Two locks, not one. get_vector_store() needs the embeddings to construct Chroma, and
# threading.Lock is not reentrant — guarding both with a single lock deadlocked the very
# first upload on a cold process (store lock held, then blocking forever on itself).
_emb_lock = threading.Lock()
_store_lock = threading.Lock()


class DocumentError(RuntimeError):
    """Raised when a file can't be read: unsupported type, encrypted, or no text."""


class OnnxMiniLMEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 served through ONNX Runtime rather than torch.

    Measured on this stack, `import torch` alone costs ~190 MB of resident memory
    before a model is loaded, and sentence-transformers pushes the process past
    520 MB — which does not fit the 512 MB free tiers this deploys to. ONNX Runtime
    ships inside chromadb already, so this swaps the backend without adding a
    dependency and without changing the model or its 384 dimensions.

    Vectors are L2-normalised so that Chroma's distance metric behaves as cosine
    similarity, matching the previous sentence-transformers configuration.
    """

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._ef = embedding_functions.ONNXMiniLM_L6_V2()

    @staticmethod
    def _normalise(vector) -> List[float]:
        values = [float(x) for x in vector]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values] if norm else values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._normalise(v) for v in self._ef(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> OnnxMiniLMEmbeddings:
    """Cached — the lock stops two concurrent requests loading the model twice."""
    global _embeddings
    if _embeddings is None:
        with _emb_lock:
            if _embeddings is None:
                _embeddings = OnnxMiniLMEmbeddings()
    return _embeddings


def get_vector_store() -> Chroma:
    """Cached singleton Chroma client."""
    global _vector_store
    if _vector_store is None:
        # Resolved before taking _store_lock: acquiring one lock while holding another
        # is what deadlocked here previously.
        embeddings = get_embeddings()
        with _store_lock:
            if _vector_store is None:
                _vector_store = Chroma(
                    collection_name=COLLECTION,
                    persist_directory=config.CHROMA_DIR,
                    embedding_function=embeddings,
                )
    return _vector_store


def clear_collection() -> None:
    global _vector_store
    try:
        get_vector_store().delete_collection()
    except Exception:
        pass  # collection may not exist yet
    _vector_store = None


def _extract_pdf(data: bytes, filename: str) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise DocumentError(f"Could not read '{filename}' as a PDF ({exc}).")

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise DocumentError(f"'{filename}' is password-protected.")

    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)

    if not text.strip():
        raise DocumentError(
            f"No text layer found in '{filename}'. It is most likely a scanned image "
            "PDF — this pipeline reads embedded text and does not run OCR."
        )
    return text


def _extract_text(data: bytes, filename: str) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except Exception:
            raise DocumentError(f"Could not decode '{filename}' as text.")

    if not text.strip():
        raise DocumentError(f"'{filename}' is empty.")
    return text


def extract(data: bytes, filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(data, filename)
    if lower.endswith(TEXT_SUFFIXES):
        return _extract_text(data, filename)
    raise DocumentError(
        f"'{filename}' is not a supported type. Accepted: {', '.join(SUPPORTED_SUFFIXES)}."
    )


def add_document(data: bytes, filename: str) -> int:
    """Extracts, chunks, and embeds one document. Returns the chunks added."""
    text = extract(data, filename)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    ).split_documents([Document(page_content=text, metadata={"source": filename})])

    get_vector_store().add_documents(chunks)
    return len(chunks)


def collection_stats() -> dict:
    """Chunk count plus a per-source breakdown, for the corpus panel."""
    try:
        data = get_vector_store().get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
    except Exception:
        return {"count": 0, "sources": []}

    counts: dict[str, int] = {}
    for meta in metadatas:
        source = (meta or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1

    return {
        "count": len(metadatas),
        "sources": [{"source": s, "chunks": n} for s, n in sorted(counts.items())],
    }
