"""Tests the graph's wiring and retry/refuse logic with fake LLMs and a fake
vector store, so no ANTHROPIC_API_KEY or network access is required.
"""
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag.graph import build_graph
from rag.schemas import CriticVerdict
from rag.state import RAGState


class FakeVectorStore:
    def similarity_search(self, query, k):
        return [Document(page_content=f"chunk for '{query}'", metadata={"source": "fake.md"})]


class FakeGenerator:
    """Returns a fixed or per-call answer; records every call it received."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        return AIMessage(content=answer)


class FakeCritic:
    """Returns a queued sequence of CriticVerdict objects, one per call."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self._verdicts.pop(0)


def make_initial_state(question="What regions are available?", max_attempts=3) -> RAGState:
    return {
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


def test_accepts_immediately_when_grounded():
    generator = FakeGenerator(["Nimbus operates four regions."])
    critic = FakeCritic([
        CriticVerdict(verdict="grounded", reasoning="Matches context.", reformulated_query=None),
    ])

    app = build_graph(FakeVectorStore(), generator_llm=generator, critic_llm=critic)
    result = app.invoke(make_initial_state())

    assert result["status"] == "answered"
    assert result["attempts"] == 1
    assert result["final_answer"] == "Nimbus operates four regions."
    assert len(result["trace"]) == 1


def test_retries_with_reformulated_query_then_succeeds():
    generator = FakeGenerator(["wrong answer", "corrected answer"])
    critic = FakeCritic([
        CriticVerdict(
            verdict="hallucinated",
            reasoning="Claim not in context.",
            reformulated_query="regions available for deployment",
        ),
        CriticVerdict(verdict="grounded", reasoning="Now supported.", reformulated_query=None),
    ])

    app = build_graph(FakeVectorStore(), generator_llm=generator, critic_llm=critic)
    result = app.invoke(make_initial_state())

    assert result["status"] == "answered"
    assert result["attempts"] == 2
    assert result["final_answer"] == "corrected answer"
    # second retrieval used the critic's reformulated query
    assert result["trace"][1]["query"] == "regions available for deployment"


def test_refuses_immediately_when_critic_offers_no_reformulation():
    generator = FakeGenerator(["a guess"])
    critic = FakeCritic([
        CriticVerdict(
            verdict="insufficient_information",
            reasoning="Knowledge base does not cover GPUs.",
            reformulated_query=None,
        ),
    ])

    app = build_graph(FakeVectorStore(), generator_llm=generator, critic_llm=critic)
    result = app.invoke(make_initial_state("Does Nimbus offer GPUs?"))

    assert result["status"] == "refused"
    # gave up after one cycle rather than burning the remaining attempts
    assert result["attempts"] == 1
    assert len(generator.calls) == 1


def test_refuses_after_max_attempts_exhausted():
    generator = FakeGenerator(["still not grounded"])
    critic = FakeCritic([
        CriticVerdict(
            verdict="insufficient_information", reasoning="Not covered.", reformulated_query="try again"
        ),
        CriticVerdict(
            verdict="insufficient_information", reasoning="Still not covered.", reformulated_query="try once more"
        ),
    ])

    app = build_graph(FakeVectorStore(), generator_llm=generator, critic_llm=critic)
    result = app.invoke(make_initial_state(max_attempts=2))

    assert result["status"] == "refused"
    assert result["attempts"] == 2
    assert "don't have enough" in result["final_answer"]
