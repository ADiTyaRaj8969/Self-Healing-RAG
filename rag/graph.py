from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from . import config
from .llm import critic_llm as default_critic_llm
from .llm import generator_llm as default_generator_llm
from .prompts import (
    CRITIC_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    build_critic_prompt,
    build_generation_prompt,
    format_docs,
)
from .schemas import CriticVerdict
from .state import RAGState


def decide_next(state: RAGState) -> str:
    """Routes after critique: accept the answer, retry retrieval, or refuse."""
    verdict: CriticVerdict = state["critic_verdict"]
    if verdict.verdict == "grounded":
        return "accept"
    if state["attempts"] >= state["max_attempts"]:
        return "refuse"
    # The critic omits a reformulation when it judges the knowledge base simply
    # doesn't cover the topic; retrying the same query would retrieve the same chunks.
    if not verdict.reformulated_query:
        return "refuse"
    return "retry"


def finalize_node(state: RAGState) -> dict:
    return {"final_answer": state["answer"], "status": "answered"}


def refuse_node(state: RAGState) -> dict:
    return {
        "final_answer": (
            "I don't have enough reliably-grounded information in the knowledge base to "
            "answer that confidently, even after reformulating the search."
        ),
        "status": "refused",
    }


def build_graph(vector_store, generator_llm=None, critic_llm=None):
    """Wires the self-healing RAG graph.

    `generator_llm` / `critic_llm` default to the provider configured by LLM_PROVIDER,
    but can be swapped for test doubles that implement `.invoke(messages)` — the
    generator returning an object with `.content`, the critic returning a
    `CriticVerdict` directly.
    """
    generator_llm = generator_llm or default_generator_llm()
    critic_llm = critic_llm or default_critic_llm()

    def retrieve_node(state: RAGState) -> dict:
        query = state["current_query"]
        docs = vector_store.similarity_search(query, k=config.TOP_K)
        return {"retrieved_docs": docs, "last_retrieval_query": query}

    def generate_node(state: RAGState) -> dict:
        context = format_docs(state["retrieved_docs"])
        messages = [
            SystemMessage(GENERATOR_SYSTEM_PROMPT),
            HumanMessage(build_generation_prompt(state["question"], context)),
        ]
        response = generator_llm.invoke(messages)
        return {"answer": response.content}

    def critique_node(state: RAGState) -> dict:
        context = format_docs(state["retrieved_docs"])
        messages = [
            SystemMessage(CRITIC_SYSTEM_PROMPT),
            HumanMessage(build_critic_prompt(state["question"], context, state["answer"])),
        ]
        verdict: CriticVerdict = critic_llm.invoke(messages)

        attempts = state["attempts"] + 1
        trace_entry = {
            "attempt": attempts,
            "query": state["current_query"],
            "answer": state["answer"],
            "verdict": verdict.verdict,
            "reasoning": verdict.reasoning,
        }
        update = {
            "critic_verdict": verdict,
            "attempts": attempts,
            "trace": state["trace"] + [trace_entry],
        }
        if verdict.verdict != "grounded" and verdict.reformulated_query:
            update["current_query"] = verdict.reformulated_query
        return update

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique",
        decide_next,
        {"accept": "finalize", "retry": "retrieve", "refuse": "refuse"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("refuse", END)

    return graph.compile()
