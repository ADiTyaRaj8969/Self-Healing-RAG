from typing import Literal, Optional

from pydantic import BaseModel, Field


class CriticVerdict(BaseModel):
    """Structured output produced by the critic node."""

    verdict: Literal["grounded", "hallucinated", "insufficient_information"] = Field(
        description=(
            "'insufficient_information' if the answer fails to actually answer the "
            "question — including any decline, hedge, or statement that the information "
            "is unavailable — or if the context plainly lacks what was asked. "
            "'hallucinated' if the answer does answer the question but contains a claim, "
            "number, name, or detail not traceable to the context. "
            "'grounded' only if the answer genuinely answers the question AND every "
            "claim in it is supported by the context."
        )
    )
    reasoning: str = Field(
        description="Brief explanation of the verdict, citing which claims were or were not supported."
    )
    reformulated_query: Optional[str] = Field(
        default=None,
        description=(
            "Only when verdict is not 'grounded': a reformulated search query, more "
            "specific or using alternate terminology, that is more likely to retrieve "
            "context supporting the ORIGINAL question. Omit if you believe the "
            "knowledge base simply does not cover this topic."
        ),
    )
