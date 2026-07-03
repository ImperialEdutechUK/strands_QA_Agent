"""OpenRouter provider-routing policy.

By DEFAULT this is now PERMISSIVE: OpenRouter may route each request to ANY
provider, with fallbacks enabled, so a request is never queued behind a small
allow-list of saturated providers (which was surfacing as sustained HTTP 429s and
minutes-long "the run is stuck" behaviour). Availability is the priority.

Everything is opt-in via env vars, so a privacy-sensitive deployment can re-tighten
the policy without touching code:

  OPENROUTER_ONLY_PROVIDERS   — comma-separated allow-list (e.g. "DeepInfra,Novita").
                                Empty (default) = no restriction, any provider.
  OPENROUTER_ALLOW_FALLBACKS  — "0" to disable provider fallbacks; default "1"
                                (enabled) so a busy provider spills to the next.
  OPENROUTER_DATA_COLLECTION  — "deny" to only use providers that contractually do
                                NOT log/retain/train on the prompt (GDPR-style
                                privacy, but a smaller pool that CAN queue), or
                                "allow" (default) to permit any provider so the
                                request is never held back.
"""

from __future__ import annotations

import os


def _csv_env(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def openrouter_provider_block() -> dict:
    """Return the `provider` payload to attach to every OpenRouter request.

    Permissive by default (any provider, fallbacks on) so requests aren't queued;
    tighten via the OPENROUTER_* env vars documented in the module docstring.
    """
    block: dict = {}

    # Allow-list is OPT-IN now. When empty we send no `only` key, so OpenRouter is
    # free to route to whatever provider can serve the model soonest.
    allowed = _csv_env("OPENROUTER_ONLY_PROVIDERS")
    if allowed:
        block["only"] = list(allowed)

    # Fallbacks ON by default: if the first-choice provider is busy/at capacity,
    # spill to the next eligible one rather than failing or waiting.
    block["allow_fallbacks"] = os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "1").strip() != "0"

    # Privacy filter is OPT-IN. Default "allow" keeps the provider pool as large as
    # possible (no queueing); set OPENROUTER_DATA_COLLECTION=deny to restrict to
    # non-logging providers if the deployment needs that guarantee.
    dc = os.environ.get("OPENROUTER_DATA_COLLECTION", "allow").strip().lower()
    if dc == "deny":
        block["data_collection"] = "deny"

    return block
