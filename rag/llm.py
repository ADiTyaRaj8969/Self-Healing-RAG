"""Provider-pluggable chat model factory.

The graph needs only two things from a provider: a chat model whose response has
`.content`, and one that can emit a `CriticVerdict` through structured output.
Everything else — retrieval, prompts, routing — is provider-agnostic.
"""
import os
import threading

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


# Constructing a chat client measured ~1.2s — more than an actual Groq round trip —
# and the graph is rebuilt whenever the corpus changes, which would pay that twice
# every time. The clients are stateless and thread-safe to share, so cache them.
_clients: dict[tuple[str, str], object] = {}
_client_lock = threading.Lock()


def _cached(kind: str, model: str):
    key = (kind, model)
    client = _clients.get(key)
    if client is None:
        with _client_lock:
            client = _clients.get(key)
            if client is None:
                client = _build(model)
                if kind == "critic":
                    # Routing reads a typed field, so the critic must return a
                    # CriticVerdict rather than prose.
                    client = client.with_structured_output(CriticVerdict)
                _clients[key] = client
    return client


def reset_clients() -> None:
    """Drop cached clients, e.g. after credentials or model config change."""
    with _client_lock:
        _clients.clear()


def generator_llm():
    return _cached("generator", config.GENERATION_MODEL)


def critic_llm():
    return _cached("critic", config.CRITIC_MODEL)
