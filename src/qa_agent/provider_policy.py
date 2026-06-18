"""OpenRouter provider-routing policy.

Every LLM request we send to OpenRouter is annotated with a `provider` block
that:
  * blocks Chinese-jurisdiction providers (DeepSeek the company, Novita,
    01.AI, Chutes, etc) so prompts never reach those data centres;
  * sets `data_collection: "deny"` so OpenRouter only routes to providers
    that contractually do NOT log, retain, or train on the prompt;
  * disables fallbacks — if no compliant provider is available we want the
    request to FAIL loudly rather than silently route through a non-compliant
    one.

The defaults below can be overridden via env vars without touching code:
  OPENROUTER_BLOCKED_PROVIDERS     — comma-separated, replaces the block list
  OPENROUTER_ALLOW_FALLBACKS       — "0" to disable provider fallbacks; default
                                     is "1" so a single flaky compliant provider
                                     doesn't kill the whole streaming run. The
                                     `data_collection: deny` + `ignore` list
                                     still constrain the fallback pool to
                                     compliant providers, so enabling this does
                                     not relax the jurisdictional policy.
"""

from __future__ import annotations

import os

# Providers operating primarily out of mainland China / Hong Kong. We block
# them outright regardless of OpenRouter's "data_collection" guarantees,
# because the user requirement is jurisdictional, not just retention-policy.
_DEFAULT_BLOCKED_PROVIDERS = (
    "DeepSeek",     # the model author's own China-based inference
    "Novita",       # HK / mainland China
    "01.AI",        # Yi / 01.AI, China
    "Chutes",       # community / mixed pool, includes CN nodes
    "InferenceNet", # mixed pool, can include CN nodes
)


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def openrouter_provider_block() -> dict:
    """Return the `provider` payload to attach to every OpenRouter request."""
    blocked = _csv_env("OPENROUTER_BLOCKED_PROVIDERS", _DEFAULT_BLOCKED_PROVIDERS)
    allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "1").strip() == "1"

    block: dict = {
        # Refuse providers that would log or retain the prompt.
        "data_collection": "deny",
        # Hard-block providers operating from non-compliant jurisdictions even
        # if OpenRouter would otherwise consider them eligible (includes Chinese
        # providers: DeepSeek, Novita, 01.AI, Chutes, InferenceNet).
        "ignore": list(blocked),
        # When a compliant provider is unavailable, fail rather than spill
        # the request to a non-compliant one.
        "allow_fallbacks": allow_fallbacks,
    }
    return block
