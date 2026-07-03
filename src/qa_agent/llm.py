import os

import httpx
from strands.models.openai import OpenAIModel

from .provider_policy import openrouter_provider_block
from .security import require_env

# Hard timeout on the ORCHESTRATION model calls (the agent's own tool-selection /
# reasoning turns), NOT the tool-internal calls (those use llm_client). Without
# this the OpenAI client falls back to its 600s default and — worse — during a
# mid-stream provider stall it simply waits, so a single hung turn made the whole
# run "process forever" once the outer wall-clock cap was removed. With an
# explicit read timeout a stalled turn RAISES promptly; invoke_with_retry then
# retries it, and if it still fails the run ends with an honest "incomplete"
# report instead of hanging. `read` is the inter-chunk gap during streaming, so
# a provider that goes silent trips it while a provider that is actively emitting
# tokens never does. Override via QA_AGENT_MODEL_* env vars.
_MODEL_CONNECT_TIMEOUT = float(os.environ.get("QA_AGENT_MODEL_CONNECT_TIMEOUT", "15"))
_MODEL_READ_TIMEOUT = float(os.environ.get("QA_AGENT_MODEL_READ_TIMEOUT", "120"))
_MODEL_MAX_RETRIES = int(os.environ.get("QA_AGENT_MODEL_MAX_RETRIES", "2"))


def build_model() -> OpenAIModel:
    api_key = require_env("OPENROUTER_API_KEY")
    # `extra_body` is passed through by the OpenAI client into the JSON request,
    # so we use it to attach OpenRouter's `provider` routing preferences (which
    # restrict the request to GDPR-jurisdiction data centres and forbid the
    # provider from logging or training on the prompt).
    return OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "default_headers": {
                "HTTP-Referer": "https://localhost",
                "X-Title": "Strands QA Agent",
            },
            # Bound every orchestration request so a stalled stream can't hang
            # the run indefinitely; the OpenAI client also retries connection
            # errors up to max_retries before raising.
            "timeout": httpx.Timeout(
                connect=_MODEL_CONNECT_TIMEOUT,
                read=_MODEL_READ_TIMEOUT,
                write=30.0,
                pool=10.0,
            ),
            "max_retries": _MODEL_MAX_RETRIES,
        },
        model_id=os.environ.get("MODEL", "deepseek/deepseek-v4-pro"),
        params={
            # Greedy decoding so the orchestrator's tool-selection and the
            # agent's reasoning are deterministic rather than guessed.
            "temperature": 0.0,
            "extra_body": {"provider": openrouter_provider_block()},
        },
    )
