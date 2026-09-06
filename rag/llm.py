"""Provider-pluggable chat model factory.

The graph needs only two things from a provider: a chat model whose response has
`.content`, and one that can emit a `CriticVerdict` through structured output.
Everything else — retrieval, prompts, routing — is provider-agnostic.
"""
import os

from . import config
from .schemas import CriticVerdict

SUPPORTED_PROVIDERS = ("groq", "anthropic")

_KEY_ENV_VAR = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class MissingCredentialsError(RuntimeError):
    """Raised when the selected provider has no usable API key configured."""


def validate_credentials() -> None:
    provider = os.getenv("LLM_PROVIDER", config.LLM_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise MissingCredentialsError(
            f"Unknown LLM_PROVIDER={provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    key = (os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY) if provider == "groq" else (os.getenv("ANTHROPIC_API_KEY") or config.ANTHROPIC_API_KEY)
    if not key:
        env_var = _KEY_ENV_VAR[provider]
        raise MissingCredentialsError(
            f"{env_var} is not set, but LLM_PROVIDER={provider}. "
            f"Add {env_var} to your .env file or Space Secrets (see .env.example)."
        )


def _build(model: str, temperature: float = 0):
    validate_credentials()

    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, temperature=temperature)


def generator_llm():
    return _build(config.GENERATION_MODEL)


def critic_llm():
    """The critic must return a CriticVerdict, not prose — routing reads a typed field."""
    return _build(config.CRITIC_MODEL).with_structured_output(CriticVerdict)
