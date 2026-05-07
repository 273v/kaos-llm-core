"""Router serialization — convert :class:`CascadeRouter` and :class:`RuleRouter`
instances to/from JSON-friendly state dicts.

Used by ``Call.get_learnable_state()`` / ``Call.set_learnable_state()`` so that
optimized programs that use a router survive ``Program.save()`` /
``Program.load()`` round-trips. This closes F5 (Router learnable-state
carry-forward) — Phase 5 descoped router persistence and Phase 6.2
(``ModelOptimizer``) was supposed to add it but didn't.

Limitations
-----------

* :class:`CascadeRouter`'s ``escalation_check`` is a Python callable. Pickling
  arbitrary callables across process boundaries is unsafe and would couple
  saved state to in-process import paths, so the predicate is **dropped on
  save**. Loading a previously-saved CascadeRouter reconstructs it with the
  default predicate (``lambda _: True``), which means every step always passes
  and the router collapses to "use the first model." Programs that rely on a
  custom escalation predicate must restore it manually after load:

  .. code-block:: python

      program.load("tuned.json")
      program.my_call._router.escalation_check = my_predicate

  This is documented loudly on every save path that hits a CascadeRouter with
  a non-default predicate so users can't be silently surprised.

* The router state schema is part of the v2 ``Program.save()`` envelope (the
  ``state`` dict's ``router`` key, ``None`` when no router is set). v1 envelopes
  loaded into a Call that has a router will leave that router untouched.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_core.router.cascade import CascadeRouter
from kaos_llm_core.router.rules import Rule, RuleRouter

logger = get_logger(__name__)

__all__ = ["router_from_state", "router_to_state"]


def router_to_state(router: Any) -> dict[str, Any] | None:
    """Serialize a router instance to a JSON-friendly dict.

    Returns ``None`` for ``router=None`` (no router) or for router types this
    module doesn't recognize. Unknown router types are logged at WARNING so
    callers can decide whether the gap matters.
    """
    if router is None:
        return None
    if isinstance(router, CascadeRouter):
        # If the user passed a non-trivial escalation_check, warn them that
        # it will not survive the round-trip. Phase 9e: identity-check the
        # module-level sentinel instead of comparing lambda bytecode (the
        # previous fragile approach broke between Python minor versions).
        from kaos_llm_core.router.cascade import _default_escalation_check

        is_default_predicate = router.escalation_check is _default_escalation_check
        if not is_default_predicate:
            logger.warning(
                "router_to_state: CascadeRouter has a custom escalation_check "
                "predicate; it cannot be serialized and will be dropped. After "
                "loading, restore it manually via "
                "``call._router.escalation_check = ...``."
            )
        return {
            "type": "cascade",
            "models": list(router.models),
        }
    if isinstance(router, RuleRouter):
        return {
            "type": "rule",
            "default_model": router.default_model,
            "rules": [
                {
                    "model": r.model,
                    "signature_name": r.signature_name,
                    "input_matches": dict(r.input_matches),
                }
                for r in router.rules
            ],
        }
    logger.warning(
        "router_to_state: unknown router type %s; not serializing. "
        "Implement router_to_state for the new type, or omit it from "
        "learnable state.",
        type(router).__name__,
    )
    return None


def router_from_state(state: dict[str, Any] | None) -> Any:
    """Reconstruct a router from a state dict produced by :func:`router_to_state`.

    Returns ``None`` if ``state`` is None or the type is unknown. Unknown
    types are logged at WARNING.
    """
    if not state:
        return None
    kind = state.get("type")
    if kind == "cascade":
        models = list(state.get("models") or [])
        if not models:
            logger.warning("router_from_state: cascade state had no models; returning None")
            return None
        # escalation_check defaults to always-pass; users restore the
        # custom predicate after load if they need one.
        return CascadeRouter(models=models)
    if kind == "rule":
        rules = [
            Rule(
                model=r["model"],
                signature_name=r.get("signature_name"),
                input_matches=dict(r.get("input_matches") or {}),
            )
            for r in (state.get("rules") or [])
        ]
        return RuleRouter(rules=rules, default_model=state.get("default_model"))
    logger.warning("router_from_state: unknown router type %r; not reconstructing.", kind)
    return None
