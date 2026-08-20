"""Local, token-free activity timeline for Codex and GPU workloads."""

from .phases import CODEX_PHASES, normalize_phase

__all__ = ["CODEX_PHASES", "normalize_phase"]
