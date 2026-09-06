from typing import List, Optional, TypedDict

from langchain_core.documents import Document

from .schemas import CriticVerdict


class RAGState(TypedDict):
    # The user's original question. Never mutated.
    question: str
    # The query actually used for the current retrieval pass. Starts equal to
    # `question`; the critic may rewrite it on retry.
    current_query: str

    retrieved_docs: List[Document]
    # The query the most recent retrieval actually ran with. Redundant with
    # `current_query` for the CLI, but lets a streaming client label each pass
    # without re-deriving the critic's rewrite logic.
    last_retrieval_query: Optional[str]
    answer: str
    critic_verdict: Optional[CriticVerdict]

    attempts: int
    max_attempts: int
    # Per-attempt record of {attempt, query, answer, verdict, reasoning}, for
    # debugging / --verbose output.
    trace: List[dict]

    final_answer: Optional[str]
    status: Optional[str]  # "answered" | "refused"
