"""OpenRouter provider-routing policy.

Every LLM request we send to OpenRouter is annotated with a `provider` block
that:
  * restricts routing to an explicit ALLOW-LIST of providers — only
    DigitalOcean, AtlasCloud and DeepInfra are permitted; OpenRouter will not
    route the request to any other provider;
  * sets `data_collection: "deny"` so the allowed providers only handle the
    request if they contractually do NOT log, retain, or train on the prompt;
  * disables fallbacks — if none of the allowed providers can serve the
    request we want it to FAIL loudly rather than silently route elsewhere.

The defaults below can be overridden via env vars without touching code:
  OPENROUTER_ONLY_PROVIDERS    — comma-separated, replaces the allow-list
  OPENROUTER_ALLOW_FALLBACKS   — "1" to re-enable provider fallbacks; default
                                 is "0" (disabled) so the request fails rather
                                 than spilling onto a provider outside the
                                 allow-list.
"""

from __future__ import annotations

import os

# The ONLY providers OpenRouter is permitted to route to. Anything not in this
# list is excluded, regardless of OpenRouter's own eligibility scoring.
_DEFAULT_ALLOWED_PROVIDERS = (
    "DigitalOcean",
    "AtlasCloud",
    "DeepInfra",
)


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def openrouter_provider_block() -> dict:
    """Return the `provider` payload to attach to every OpenRouter request."""
    allowed = _csv_env("OPENROUTER_ONLY_PROVIDERS", _DEFAULT_ALLOWED_PROVIDERS)
    # Fallbacks are OFF by default: if no allowed provider can serve the
    # request, fail rather than route to a provider outside the allow-list.
    allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "0").strip() == "1"

    block: dict = {
        # Hard allow-list: route ONLY to these providers, nothing else.
        "only": list(allowed),
        # Of the allowed providers, refuse any that would log or retain the prompt.
        "data_collection": "deny",
        # No spilling onto providers outside the allow-list.
        "allow_fallbacks": allow_fallbacks,
    }
    return block
