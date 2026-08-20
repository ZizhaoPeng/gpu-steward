"""Stable phase vocabulary shared by collectors, aggregation, and the UI."""

from typing import Tuple

from ..errors import CommandError


CODEX_PHASES: Tuple[str, ...] = (
    "research",
    "review",
    "analysis",
    "implement",
    "test",
    "operate",
    "active-unspecified",
    "waiting-tool",
    "waiting-user",
    "suspected-stall",
    "idle",
)


def normalize_phase(value: str) -> str:
    """Return a canonical phase or fail instead of inventing a label."""

    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized not in CODEX_PHASES:
        raise CommandError(
            "unknown Codex phase {!r}; expected one of: {}".format(
                value, ", ".join(CODEX_PHASES)
            )
        )
    return normalized
