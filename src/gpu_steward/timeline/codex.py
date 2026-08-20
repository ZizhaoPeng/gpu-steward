"""Fail-closed ingestion for official Codex lifecycle hook JSON.

Hooks are treated as a transport boundary, not as a transcript.  Only a
small allow-list of lifecycle metadata is extracted.  Prompt/response text,
command arguments, environment variables, paths, and arbitrary hook payloads
are discarded before the value reaches :mod:`timeline.store`.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import re
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from .phases import normalize_phase
from .store import (
    EVENT_SOURCES,
    TimelineSchemaError,
    TimelineStore,
    _normalize_kind,
    _normalize_source,
    _project_name,
)


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}$")
_SAFE_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")

# The category is a stable display label, never the original tool name.  The
# values intentionally contain no command/path data.
TOOL_CATEGORIES = (
    "shell",
    "python",
    "ssh",
    "file",
    "browser",
    "http",
    "gpu",
    "other",
)


def _first(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _nested(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("data", "metadata", "context"):
        value = payload.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _parse_timestamp(value: Any, now: Optional[float] = None) -> float:
    if value is None:
        return time.time() if now is None else float(now)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            result = float(text)
        except ValueError:
            try:
                normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
                parsed = _datetime.datetime.fromisoformat(normalized)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=_datetime.timezone.utc)
                result = parsed.timestamp()
            except (TypeError, ValueError, OverflowError) as exc:
                raise TimelineSchemaError("occurred_at must be Unix seconds or ISO-8601") from exc
    else:
        raise TimelineSchemaError("occurred_at must be Unix seconds or ISO-8601")
    if result != result or result in (float("inf"), float("-inf")) or abs(result) > 10 ** 12:
        raise TimelineSchemaError("occurred_at is outside the supported Unix-time range")
    return result


def _safe_hook_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value and _SAFE_ID_RE.match(value) else None


def _event_name(payload: Mapping[str, Any], nested: Mapping[str, Any]) -> Any:
    value = _first(
        payload,
        "hook_event_name",
        "hook_event",
        "event_name",
        "event_type",
        "event",
        "kind",
        "type",
    )
    if value is None:
        value = _first(
            nested,
            "hook_event_name",
            "hook_event",
            "event_name",
            "event_type",
            "kind",
            "type",
        )
    if isinstance(value, Mapping):
        value = _first(value, "name", "kind", "type")
    return value


def _map_tool_category(value: Any) -> str:
    """Convert a tool identifier into a safe, finite category vocabulary."""

    if value is None:
        return ""
    if not isinstance(value, str):
        raise TimelineSchemaError("tool_category/tool_name must be a string")
    text = value.strip().lower()
    if not text:
        return ""
    # A caller-supplied category is accepted only if it is already a safe
    # category.  Raw tool names are mapped by prefix/known aliases; the raw
    # value never leaves this function.
    if _SAFE_CATEGORY_RE.match(text) and text in TOOL_CATEGORIES:
        return text
    if text in ("bash", "zsh", "sh", "shell", "terminal", "exec", "command", "run"):
        return "shell"
    if text.startswith("python") or text in ("py", "jupyter"):
        return "python"
    if text in ("ssh", "remote", "remote-shell") or text.startswith("ssh-"):
        return "ssh"
    if text in ("read", "write", "edit", "apply-patch", "apply_patch", "file", "filesystem"):
        return "file"
    if text in ("browser", "browser-use", "computer", "computer-use", "playwright"):
        return "browser"
    if text in ("http", "fetch", "web", "web-search"):
        return "http"
    if text in ("gpu", "nvidia-smi", "collector", "sample"):
        return "gpu"
    return "other"


def _bool_or_default(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TimelineSchemaError("tool_active must be boolean")
    return value


def normalize_hook(payload: Mapping[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    """Return only safe canonical metadata extracted from one hook object.

    The returned mapping is intentionally suitable for passing directly to
    :meth:`TimelineStore.record_codex_event`; it contains no original payload
    and no prompt/response/command/environment fields.
    """

    if not isinstance(payload, Mapping):
        raise TimelineSchemaError("Hook payload must be a JSON object")
    nested = _nested(payload)
    raw_kind = _event_name(payload, nested)
    if raw_kind is None:
        raise TimelineSchemaError("Hook payload has no recognized event kind")
    kind = _normalize_kind(raw_kind)
    timestamp = _parse_timestamp(
        _first(payload, "occurred_at", "timestamp", "ts", "time")
        if _first(payload, "occurred_at", "timestamp", "ts", "time") is not None
        else _first(nested, "occurred_at", "timestamp", "ts", "time"),
        now,
    )

    session_id = _first(payload, "session_id", "sessionId", "session")
    if session_id is None:
        session_id = _first(nested, "session_id", "sessionId", "session")
    turn_id = _first(payload, "turn_id", "turnId", "turn")
    if turn_id is None:
        turn_id = _first(nested, "turn_id", "turnId", "turn")
    # Missing identifiers are represented by stable placeholders and then
    # salted/hashed by TimelineStore.  The raw values are never returned.
    session_id = "unknown-session" if session_id is None else str(session_id)
    turn_id = "unknown-turn" if turn_id is None else str(turn_id)

    project_value = _first(payload, "project", "project_name", "cwd", "working_directory")
    if project_value is None:
        project_value = _first(nested, "project", "project_name", "cwd", "working_directory")
    project = _project_name(project_value)

    raw_phase = _first(payload, "phase", "declared_phase")
    if raw_phase is None:
        raw_phase = _first(nested, "phase", "declared_phase")
    if kind == "pre-tool":
        phase = "waiting-tool"
    elif kind in ("stop", "turn-end"):
        phase = "waiting-user"
    elif raw_phase is None:
        phase = "active-unspecified"
    else:
        try:
            phase = normalize_phase(raw_phase)
        except Exception as exc:
            # ``phases.normalize_phase`` is also used by the public CLI and
            # raises CommandError there.  Hook ingestion exposes one stable
            # schema boundary, so normalize that implementation detail to a
            # timeline schema error.
            raise TimelineSchemaError("unknown Codex phase {!r}".format(raw_phase)) from exc

    source_value = _first(payload, "source", "phase_source")
    if source_value is None:
        source_value = _first(nested, "source", "phase_source")
    if source_value is None:
        source = "declared" if kind == "phase" and raw_phase is not None else "hook-rule"
    else:
        source = _normalize_source(source_value)
    confidence_value = _first(payload, "confidence", "phase_confidence")
    if confidence_value is None:
        confidence_value = _first(nested, "confidence", "phase_confidence")
    if confidence_value is None:
        confidence = 1.0 if source == "declared" else (0.5 if source == "inferred" else 0.8)
    else:
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError) as exc:
            raise TimelineSchemaError("confidence must be a finite number") from exc
        if confidence != confidence or confidence in (float("inf"), float("-inf")):
            raise TimelineSchemaError("confidence must be a finite number")
        if not 0.0 <= confidence <= 1.0:
            raise TimelineSchemaError("confidence must be between 0 and 1")

    raw_tool = _first(payload, "tool_category", "tool", "tool_name", "toolName")
    if raw_tool is None:
        raw_tool = _first(nested, "tool_category", "tool", "tool_name", "toolName")
    tool_category = _map_tool_category(raw_tool)
    raw_active = _first(payload, "tool_active")
    if raw_active is None:
        raw_active = _first(nested, "tool_active")
    if kind == "pre-tool":
        tool_active = True
    elif kind == "post-tool":
        tool_active = False
    else:
        tool_active = _bool_or_default(raw_active, False)

    event_id = _safe_hook_id(_first(payload, "event_id", "eventId", "id"))
    if event_id is None:
        event_id = _safe_hook_id(_first(nested, "event_id", "eventId", "id"))
    if event_id is None:
        # Use only safe canonical fields.  No raw hook object is hashed or
        # persisted, because it may contain prompt/response text.
        safe = {
            "occurred_at": timestamp,
            "session_id": session_id,
            "turn_id": turn_id,
            "project": project,
            "kind": kind,
            "phase": phase,
            "source": source,
            "confidence": confidence,
            "tool_category": tool_category,
            "tool_active": tool_active,
        }
        event_id = "hook-" + hashlib.sha256(
            json.dumps(safe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return {
        "event_id": event_id,
        "occurred_at": timestamp,
        "session_id": session_id,
        "turn_id": turn_id,
        "project": project,
        "kind": kind,
        "phase": phase,
        "source": source,
        "confidence": confidence,
        "tool_category": tool_category,
        "tool_active": tool_active,
    }


def ingest_hook(
    payload: Mapping[str, Any],
    store: TimelineStore,
    now: Optional[float] = None,
) -> str:
    """Normalize one official hook object and append it idempotently."""

    if not isinstance(store, TimelineStore):
        raise TimelineSchemaError("store must be a TimelineStore")
    safe = normalize_hook(payload, now=now)
    return store.record_codex_event(**safe)


def ingest_codex_hook(
    store: TimelineStore,
    payload: Mapping[str, Any],
    now: Optional[float] = None,
) -> str:
    """Argument-order alias convenient for hook runners."""

    return ingest_hook(payload, store, now=now)


class CodexHookIngestor:
    """Small state-free adapter for a hook process reading JSON objects."""

    def __init__(self, store: TimelineStore):
        if not isinstance(store, TimelineStore):
            raise TimelineSchemaError("store must be a TimelineStore")
        self.store = store

    def ingest(self, payload: Mapping[str, Any], now: Optional[float] = None) -> str:
        return ingest_hook(payload, self.store, now=now)

    def ingest_line(self, line: str, now: Optional[float] = None) -> str:
        if not isinstance(line, str):
            raise TimelineSchemaError("Hook line must be text")
        try:
            payload = json.loads(line)
        except (TypeError, ValueError) as exc:
            raise TimelineSchemaError("Hook input is not valid JSON") from exc
        return self.ingest(payload, now=now)


process_hook = ingest_hook
handle_hook = ingest_hook


__all__ = [
    "CodexHookIngestor",
    "TOOL_CATEGORIES",
    "handle_hook",
    "ingest_codex_hook",
    "ingest_hook",
    "normalize_hook",
    "process_hook",
]
