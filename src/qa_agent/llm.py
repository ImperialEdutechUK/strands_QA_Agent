import os

from strands.models.openai import OpenAIModel

from .provider_policy import openrouter_provider_block
from .security import require_env


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
        },
        model_id=os.environ.get("MODEL", "deepseek/deepseek-v3.2"),
        params={
            "temperature": 0.2,
            "extra_body": {"provider": openrouter_provider_block()},
        },
    )
