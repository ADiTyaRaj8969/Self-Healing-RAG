import io

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config

# One corpus, populated entirely by the user's own uploads.
COLLECTION = "docs"

TEXT_SUFFIXES = (".txt", ".md", ".markdown")
SUPPORTED_SUFFIXES = (".pdf",) + TEXT_SUFFIXES

_embeddings: HuggingFaceEmbeddings | None = None


class DocumentError(RuntimeError):
    """Raised when a file can't be read: unsupported type, encrypted, or no text."""


def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached — loading BGE-M3 costs several seconds and ~2 GB, so do it once."""
    global _embeddings
    if _embeddings is None:
        # BGE models are trained/evaluated with cosine similarity on normalized
        # embeddings; Chroma's default distance metric assumes that too.
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=config.CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


def clear_collection() -> None:
    try:
        get_vector_store().delete_collection()
    except Exception:
        pass  # collection may not exist yet


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
