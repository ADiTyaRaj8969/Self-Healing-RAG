import io
import math
import os
import threading
from functools import cached_property
from typing import List

try:  # module path is internal to chromadb; fall back to the public re-export
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
except ImportError:  # pragma: no cover - depends on chromadb version
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

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

# Ingestion is serialised. Embedding peaks around 350 MB for a large document, so two
# concurrent uploads would exceed a 512 MB container even though each fits alone.
# Queuing makes a second upload wait; it does not make it fail.
_ingest_lock = threading.Lock()


class DocumentError(RuntimeError):
    """Raised when a file can't be read: unsupported type, encrypted, or no text."""


def _available_cpus() -> int:
    """Effective CPUs for this process, honouring container quota.

    os.cpu_count() reports the *host's* cores, so inside a container limited to a
    fraction of a CPU it over-reports badly (12 on a 0.1-CPU box). ONNX Runtime sizes
    its thread pool from that number by default, so it spawns a dozen threads to share
    a tenth of a core and spends most of its time context-switching. Reading the cgroup
    quota gives the real budget.
    """
    for quota_file, period_file in (
        ("/sys/fs/cgroup/cpu.max", None),  # cgroup v2: "<quota> <period>" or "max <period>"
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            with open(quota_file) as fh:
                raw = fh.read().split()
            if period_file is None:
                quota, period = raw[0], raw[1]
            else:
                quota = raw[0]
                with open(period_file) as fh:
                    period = fh.read().strip()
            if quota in ("max", "-1"):
                break
            cpus = float(quota) / float(period)
            if cpus > 0:
                return max(1, int(cpus))
        except (OSError, ValueError, IndexError):
            continue
    return max(1, os.cpu_count() or 1)


class _ThreadPinnedONNXMiniLM(ONNXMiniLM_L6_V2):
    """chroma's ONNX embedder with its thread pool sized to the real CPU budget.

    Upstream builds SessionOptions without setting intra_op_num_threads, so ONNX
    Runtime defaults to one thread per detected core. Everything else here mirrors
    the parent implementation.
    """

    @cached_property
    def model(self):  # type: ignore[override]
        so = self.ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = _available_cpus()
        so.inter_op_num_threads = 1
        so.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        # ONNX Runtime's arena allocator grows to the largest batch it has seen and
        # never returns it to the OS, so RSS stayed ~360 MB above baseline after one
        # upload. Disabling the arena trades a little speed for memory that is actually
        # released — necessary to stay inside a 512 MB container.
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        return self.ort.InferenceSession(
            os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )


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
        self._ef = _ThreadPinnedONNXMiniLM()

    @staticmethod
    def _normalise(vector) -> List[float]:
        values = [float(x) for x in vector]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values] if norm else values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed in bounded batches so peak memory doesn't scale with document size.

        Passing every chunk in one call made ONNX allocate activations for the whole
        batch at once: a 163-chunk document peaked at 739 MB, over the 512 MB limit,
        and the container was killed mid-upload. Batching keeps the peak flat — the
        returned vectors themselves are tiny (384 floats each).
        """
        out: List[List[float]] = []
        size = max(1, config.EMBED_BATCH_SIZE)
        for start in range(0, len(texts), size):
            for vector in self._ef(texts[start : start + size]):
                out.append(self._normalise(vector))
        return out

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

    # Inserted in batches for the same reason embedding is: one add_documents() call
    # holds every chunk, its vector and its metadata in memory simultaneously.
    store = get_vector_store()
    size = max(1, config.EMBED_BATCH_SIZE)
    with _ingest_lock:
        for start in range(0, len(chunks), size):
            store.add_documents(chunks[start : start + size])

    return len(chunks)


def collection_count() -> int:
    """Just the chunk count, without pulling every row's metadata.

    The /api/ask guard only needs to know whether the corpus is empty. Using
    collection_stats() for that fetched all metadata on every question, which on a
    CPU-throttled host cost seconds before the pipeline even started.
    """
    try:
        return get_vector_store()._collection.count()
    except Exception:
        return 0


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
