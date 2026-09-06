import argparse
import sys
from pathlib import Path

from rag import config
from rag.graph import build_graph
from rag.ingest import (
    SUPPORTED_SUFFIXES,
    DocumentError,
    add_document,
    clear_collection,
    collection_stats,
    get_vector_store,
)
from rag.llm import MissingCredentialsError, validate_credentials
from rag.state import RAGState


def cmd_add(args: argparse.Namespace) -> None:
    total = 0
    for path_str in args.paths:
        path = Path(path_str)
        if not path.is_file():
            print(f"skipped  {path} — not a file")
            continue
        try:
            added = add_document(path.read_bytes(), path.name)
            total += added
            print(f"added    {path.name} — {added} chunks")
        except DocumentError as exc:
            print(f"failed   {exc}")

    stats = collection_stats()
    print(f"\nCorpus now holds {stats['count']} chunks from {len(stats['sources'])} file(s). "
          f"Added {total} this run.")


def cmd_list(args: argparse.Namespace) -> None:
    stats = collection_stats()
    if not stats["sources"]:
        print("Corpus is empty. Add documents with: python main.py add <file>...")
        return
    for s in stats["sources"]:
        print(f"{s['chunks']:>5}  {s['source']}")
    print(f"\n{stats['count']} chunks from {len(stats['sources'])} file(s).")


def cmd_clear(args: argparse.Namespace) -> None:
    clear_collection()
    print("Corpus cleared.")


def cmd_ask(args: argparse.Namespace) -> None:
    try:
        validate_credentials()
    except MissingCredentialsError as exc:
        raise SystemExit(str(exc))

    if collection_stats()["count"] == 0:
        raise SystemExit(
            "Corpus is empty. Add documents first: python main.py add <file>..."
        )

    app = build_graph(get_vector_store())

    initial_state: RAGState = {
        "question": args.question,
        "current_query": args.question,
        "retrieved_docs": [],
        "last_retrieval_query": None,
        "answer": "",
        "critic_verdict": None,
        "attempts": 0,
        "max_attempts": args.max_attempts,
        "trace": [],
        "final_answer": None,
        "status": None,
    }

    result = app.invoke(initial_state)

    if args.verbose:
        for step in result["trace"]:
            print(f"--- Attempt {step['attempt']} ---")
            print(f"Query:     {step['query']}")
            print(f"Answer:    {step['answer']}")
            print(f"Verdict:   {step['verdict']}")
            print(f"Reasoning: {step['reasoning']}")
            print()

    print("=" * 60)
    print(f"STATUS: {result['status']}  (attempts used: {result['attempts']})")
    print("-" * 60)
    print(result["final_answer"])


def main() -> None:
    # Model output routinely contains characters (non-breaking hyphens, dashes) that
    # the default Windows console codepage can't encode, which crashes print().
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Self-healing RAG pipeline (LangGraph)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser(
        "add", help=f"Embed documents into the corpus ({', '.join(SUPPORTED_SUFFIXES)})"
    )
    p_add.add_argument("paths", nargs="+", help="Paths to documents")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="Show what's currently in the corpus")
    p_list.set_defaults(func=cmd_list)

    p_clear = sub.add_parser("clear", help="Delete every document from the corpus")
    p_clear.set_defaults(func=cmd_clear)

    p_ask = sub.add_parser("ask", help="Ask a question against the corpus")
    p_ask.add_argument("question", type=str, help="The question to ask")
    p_ask.add_argument(
        "--max-attempts", type=int, default=config.MAX_ATTEMPTS,
        help="Max retrieve/generate/critique cycles before refusing (default: from .env)",
    )
    p_ask.add_argument(
        "--verbose", action="store_true",
        help="Print every attempt's query, answer, and critic verdict",
    )
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
