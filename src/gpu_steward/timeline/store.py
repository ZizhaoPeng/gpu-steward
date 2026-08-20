"""Private, append-only storage for the GPU Steward activity timeline.

The timeline deliberately does not reuse :mod:`gpu_steward.state`.  Queue
state contains commands, working directories, and lease information which are
useful to the scheduler but are not safe timeline payloads.  This module only
accepts the small, allow-listed metadata described by the frozen timeline
contract.

All timestamps are UTC Unix seconds.  Event and sample rows are protected by
SQLite triggers in addition to the append-only Python API.  Overrides are
append-only records and are resolved by ``created_at``/``effective_from`` at
read time; the original observation is never edited or removed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .phases import CODEX_PHASES, normalize_phase


SCHEMA_VERSION = 1
DEFAULT_DB_PATH = "~/.gpu-steward/timeline.sqlite3"

GPU_STATES: Tuple[str, ...] = (
    "training",
    "managed-other",
    "external",
    "reserved",
    "idle",
    "disabled",
    "unknown",
)

EVENT_SOURCES: Tuple[str, ...] = ("declared", "hook-rule", "inferred")
OVERRIDE_TARGETS: Tuple[str, ...] = ("codex_event", "gpu_sample")

# Labels are deliberately much narrower than arbitrary text.  In particular,
# this rejects newlines/control bytes and prevents a caller from smuggling a
# prompt, response, shell command, or path into a display field.
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/@+\-]{0,127}$")
_SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+\-]{0,63}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}$")
_SAFE_TOOL_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


class TimelineError(ValueError):
    """Expected, fail-closed timeline input/storage error."""


class TimelineSchemaError(TimelineError):
    """Input did not satisfy the frozen timeline schema."""


class TimelineConflictError(TimelineError):
    """An idempotency key was reused for different immutable content."""


def _finite_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TimelineSchemaError("{} must be a finite number".format(field)) from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise TimelineSchemaError("{} must be a finite number".format(field))
    return result


def _timestamp(value: Any, field: str) -> float:
    result = _finite_number(value, field)
    # Negative Unix seconds are useful for deterministic epoch tests and are
    # valid Unix time, so only reject implausibly large values.
    if abs(result) > 10 ** 12:
        raise TimelineSchemaError("{} is outside the supported Unix-time range".format(field))
    return result


def _safe_label(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise TimelineSchemaError("{} is required".format(field))
    if not isinstance(value, str):
        raise TimelineSchemaError("{} must be a string".format(field))
    value = value.strip()
    if not value and allow_empty:
        return ""
    if not value or len(value) > 128 or not _SAFE_LABEL_RE.match(value):
        raise TimelineSchemaError("{} contains disallowed text".format(field))
    return value


def _safe_id(value: Any, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TimelineSchemaError("{} must be a string or integer".format(field))
    value = str(value).strip()
    if not value or len(value) > 256 or not _SAFE_ID_RE.match(value):
        raise TimelineSchemaError("{} contains disallowed text".format(field))
    return value


def _project_name(value: Any) -> str:
    """Map a project/cwd value to a basename-like display label.

    Hook payloads often supply ``cwd``.  We intentionally retain only the
    final component and never store the path itself.  Alias-style project
    names are accepted unchanged when they contain no path separator.
    """

    if value is None or not isinstance(value, str):
        return "unknown"
    raw = value.strip()
    if not raw:
        return "unknown"
    # Both POSIX and Windows separators are treated as paths.  Do not retain
    # a leading ``..`` or any parent path component.
    raw = raw.replace("\\", "/").rstrip("/")
    name = raw.rsplit("/", 1)[-1] or "unknown"
    if name in (".", ".."):
        return "unknown"
    try:
        return _safe_label(name, "project")
    except TimelineSchemaError:
        return "unknown"


def _basename(value: Any, field: str, *, allow_empty: bool = True) -> str:
    if value is None:
        return "" if allow_empty else _safe_label(value, field)
    if not isinstance(value, str):
        raise TimelineSchemaError("{} must be a string".format(field))
    value = value.strip().replace("\\", "/").rstrip("/")
    result = value.rsplit("/", 1)[-1]
    if not result and allow_empty:
        return ""
    return _safe_label(result, field)


def _json_value(value: Any, field: str) -> str:
    """Serialize a scalar override value without accepting arbitrary blobs."""

    if isinstance(value, (dict, list, tuple)):
        raise TimelineSchemaError("{} must be a scalar".format(field))
    if isinstance(value, bool):
        normalized: Any = value
    elif isinstance(value, (int, float, str)) or value is None:
        normalized = value
    else:
        raise TimelineSchemaError("{} has unsupported type".format(field))
    if isinstance(normalized, float) and (
        normalized != normalized or normalized in (float("inf"), float("-inf"))
    ):
        raise TimelineSchemaError("{} must be finite".format(field))
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > 512:
        raise TimelineSchemaError("{} is too long".format(field))
    return encoded


def _stable_hash(value: Any, salt: bytes) -> str:
    if value is None:
        value = "unknown"
    # The input is used only transiently; only a short salted digest reaches
    # SQLite.  The salt itself is per local timeline DB and never exported.
    raw = str(value).encode("utf-8", "strict")
    return hashlib.sha256(salt + b"\0" + raw).hexdigest()[:16]


def _canonical_safe_payload(values: Mapping[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def default_db_path() -> str:
    """Return the configured private timeline path without creating it."""

    configured = os.environ.get("GPU_STEWARD_TIMELINE_DB") or DEFAULT_DB_PATH
    return os.path.abspath(os.path.expanduser(configured))


def _normalize_gpu_state(value: Any) -> str:
    if not isinstance(value, str):
        raise TimelineSchemaError("state must be a string")
    result = value.strip().lower().replace("_", "-")
    if result not in GPU_STATES:
        raise TimelineSchemaError(
            "unknown GPU state {!r}; expected one of: {}".format(value, ", ".join(GPU_STATES))
        )
    return result


def _normalize_source(value: Any) -> str:
    if not isinstance(value, str):
        raise TimelineSchemaError("source must be a string")
    result = value.strip().lower().replace("_", "-")
    if result not in EVENT_SOURCES:
        raise TimelineSchemaError(
            "unknown event source {!r}; expected one of: {}".format(
                value, ", ".join(EVENT_SOURCES)
            )
        )
    return result


def _normalize_phase(value: Any) -> str:
    try:
        return normalize_phase(value)
    except Exception as exc:
        raise TimelineSchemaError("unknown Codex phase {!r}".format(value)) from exc


def _normalize_kind(value: Any) -> str:
    if not isinstance(value, str):
        raise TimelineSchemaError("kind must be a string")
    raw = value.strip()
    aliases = {
        "SessionStart": "session-start",
        "sessionStart": "session-start",
        "session_start": "session-start",
        "session-start": "session-start",
        "UserPromptSubmit": "user-prompt",
        "user_prompt_submit": "user-prompt",
        "user-prompt-submit": "user-prompt",
        "user-prompt": "user-prompt",
        "PreToolUse": "pre-tool",
        "pre_tool_use": "pre-tool",
        "pre-tool-use": "pre-tool",
        "pre-tool": "pre-tool",
        "PostToolUse": "post-tool",
        "post_tool_use": "post-tool",
        "post-tool-use": "post-tool",
        "post-tool": "post-tool",
        "Stop": "stop",
        "stop": "stop",
        "SessionEnd": "stop",
        "sessionEnd": "stop",
        "session_end": "stop",
        "session-end": "stop",
        "phase": "phase",
        "Phase": "phase",
        "turn-start": "turn-start",
        "turn_start": "turn-start",
        "TurnStart": "turn-start",
        "turn-end": "turn-end",
        "turn_end": "turn-end",
        "TurnEnd": "turn-end",
    }
    if raw in aliases:
        return aliases[raw]
    # Accept case-insensitive canonical forms, but do not invent unknown
    # kinds.  This is intentionally a finite vocabulary.
    lowered = raw.lower().replace("_", "-")
    if lowered in ("session-start", "user-prompt", "pre-tool", "post-tool", "stop", "phase", "turn-start", "turn-end"):
        return lowered
    raise TimelineSchemaError("unknown Codex event kind {!r}".format(value))


def _normalize_override_target(value: Any) -> str:
    if not isinstance(value, str):
        raise TimelineSchemaError("target_type must be a string")
    aliases = {"codex": "codex_event", "event": "codex_event", "gpu": "gpu_sample", "sample": "gpu_sample"}
    result = aliases.get(value.strip().lower(), value.strip().lower())
    if result not in OVERRIDE_TARGETS:
        raise TimelineSchemaError("unknown override target type {!r}".format(value))
    return result


class TimelineStore:
    """Private SQLite timeline store.

    ``path`` defaults to ``~/.gpu-steward/timeline.sqlite3``.  Supplying a
    ``salt`` is useful for deterministic tests; production callers should let
    the store create and persist a random local salt.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        salt: Optional[Any] = None,
        db_path: Optional[str] = None,
    ):
        configured = path or db_path or os.environ.get("GPU_STEWARD_TIMELINE_DB") or DEFAULT_DB_PATH
        self._in_memory = configured == ":memory:"
        self.path = ":memory:" if self._in_memory else os.path.abspath(os.path.expanduser(configured))
        if not self._in_memory:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, mode=0o700, exist_ok=True)
                try:
                    os.chmod(parent, 0o700)
                except OSError:
                    pass
            self._ensure_private_file(self.path)
        self._lock = threading.RLock()
        try:
            self.connection = sqlite3.connect(
                ":memory:" if self._in_memory else self.path,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA busy_timeout = 30000")
            self.initialize()
            stored = self._metadata("salt")
            if stored:
                self._salt = bytes.fromhex(stored)
            else:
                if salt is None:
                    generated = secrets.token_bytes(32)
                elif isinstance(salt, bytes):
                    generated = salt
                else:
                    generated = str(salt).encode("utf-8", "strict")
                if not generated:
                    raise TimelineSchemaError("salt must not be empty")
                self._salt = generated
                self.connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('salt',?)", (generated.hex(),)
                )
        except sqlite3.Error as exc:
            try:
                self.connection.close()
            except Exception:
                pass
            raise TimelineError("cannot initialize timeline database: {}".format(exc)) from exc

    @staticmethod
    def _ensure_private_file(path: str) -> None:
        flags = os.O_CREAT | os.O_RDWR
        try:
            fd = os.open(path, flags, 0o600)
            os.close(fd)
            os.chmod(path, 0o600)
        except OSError as exc:
            raise TimelineError("cannot create private timeline file {}: {}".format(path, exc)) from exc

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> "TimelineStore":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _metadata(self, key: str) -> Optional[str]:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def initialize(self) -> None:
        """Create the isolated schema and append-only guards idempotently."""

        try:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS codex_events (
                    event_id TEXT PRIMARY KEY,
                    occurred_at REAL NOT NULL,
                    session_hash TEXT NOT NULL,
                    turn_hash TEXT NOT NULL,
                    project TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    tool_category TEXT NOT NULL DEFAULT '',
                    tool_active INTEGER NOT NULL CHECK(tool_active IN (0,1)),
                    inserted_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS codex_events_time
                    ON codex_events(occurred_at, session_hash, event_id);
                CREATE TABLE IF NOT EXISTS gpu_samples (
                    sample_id TEXT PRIMARY KEY,
                    sampled_at REAL NOT NULL,
                    host TEXT NOT NULL,
                    gpu_index INTEGER,
                    gpu_uuid_short TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL,
                    task_name TEXT NOT NULL DEFAULT '',
                    attribution TEXT NOT NULL DEFAULT '',
                    process_basename TEXT NOT NULL DEFAULT '',
                    pid INTEGER,
                    inserted_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS gpu_samples_time
                    ON gpu_samples(sampled_at, host, gpu_index, sample_id);
                CREATE TABLE IF NOT EXISTS timeline_overrides (
                    override_id TEXT PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    field TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    effective_from REAL NOT NULL,
                    effective_until REAL,
                    reason TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS timeline_overrides_lookup
                    ON timeline_overrides(target_type, target_id, field, effective_from, created_at);
                INSERT OR IGNORE INTO metadata(key,value)
                    VALUES ('schema_version','1');
                CREATE TRIGGER IF NOT EXISTS codex_events_immutable_update
                    BEFORE UPDATE ON codex_events BEGIN
                        SELECT RAISE(ABORT, 'codex events are immutable');
                    END;
                CREATE TRIGGER IF NOT EXISTS codex_events_immutable_delete
                    BEFORE DELETE ON codex_events BEGIN
                        SELECT RAISE(ABORT, 'codex events are immutable');
                    END;
                CREATE TRIGGER IF NOT EXISTS gpu_samples_immutable_update
                    BEFORE UPDATE ON gpu_samples BEGIN
                        SELECT RAISE(ABORT, 'gpu samples are immutable');
                    END;
                CREATE TRIGGER IF NOT EXISTS gpu_samples_immutable_delete
                    BEFORE DELETE ON gpu_samples BEGIN
                        SELECT RAISE(ABORT, 'gpu samples are immutable');
                    END;
                CREATE TRIGGER IF NOT EXISTS timeline_overrides_immutable_update
                    BEFORE UPDATE ON timeline_overrides BEGIN
                        SELECT RAISE(ABORT, 'timeline overrides are immutable');
                    END;
                CREATE TRIGGER IF NOT EXISTS timeline_overrides_immutable_delete
                    BEFORE DELETE ON timeline_overrides BEGIN
                        SELECT RAISE(ABORT, 'timeline overrides are immutable');
                    END;
                """
            )
            if not self._in_memory:
                self._ensure_private_file(self.path)
        except sqlite3.Error as exc:
            raise TimelineError("cannot initialize timeline schema: {}".format(exc)) from exc

    @property
    def schema_version(self) -> int:
        raw = self._metadata("schema_version")
        try:
            return int(raw or "0")
        except ValueError as exc:
            raise TimelineError("invalid timeline schema version") from exc

    def _event_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "occurred_at": float(row["occurred_at"]),
            "session_hash": row["session_hash"],
            "turn_hash": row["turn_hash"],
            "project": row["project"],
            "kind": row["kind"],
            "phase": row["phase"],
            "source": row["source"],
            "confidence": float(row["confidence"]),
            "tool_category": row["tool_category"],
            "tool_active": bool(row["tool_active"]),
        }

    def _sample_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "sample_id": row["sample_id"],
            "sampled_at": float(row["sampled_at"]),
            "host": row["host"],
            "gpu_index": None if row["gpu_index"] is None else int(row["gpu_index"]),
            "gpu_uuid_short": row["gpu_uuid_short"],
            "state": row["state"],
            "task_name": row["task_name"],
            "attribution": row["attribution"],
            "process_basename": row["process_basename"],
            "pid": None if row["pid"] is None else int(row["pid"]),
        }

    def _override_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        try:
            value = json.loads(row["value_json"])
        except (TypeError, ValueError) as exc:
            raise TimelineError("invalid stored override value") from exc
        return {
            "override_id": row["override_id"],
            "target_type": row["target_type"],
            "target_id": row["target_id"],
            "field": row["field"],
            "value": value,
            "created_at": float(row["created_at"]),
            "effective_from": float(row["effective_from"]),
            "effective_until": None
            if row["effective_until"] is None
            else float(row["effective_until"]),
            "reason": row["reason"],
        }

    def record_codex_event(
        self,
        event_id: Optional[str],
        occurred_at: Any,
        session_id: Any,
        turn_id: Any,
        project: Any,
        kind: Any,
        phase: Any = "active-unspecified",
        source: Any = "hook-rule",
        confidence: Any = 1.0,
        tool_category: Any = "",
        tool_active: Any = False,
    ) -> str:
        """Append one allow-listed Codex event and return its id.

        A missing ``event_id`` is deterministically generated from safe
        metadata.  Replaying the same id and content is a no-op; reusing an
        id for different content raises :class:`TimelineConflictError`.
        """

        timestamp = _timestamp(occurred_at, "occurred_at")
        normalized_kind = _normalize_kind(kind)
        # Event kinds have a safe default phase; an explicit phase is always
        # validated, including for Stop/PreToolUse records.
        normalized_phase = _normalize_phase(phase)
        normalized_source = _normalize_source(source)
        normalized_confidence = _finite_number(confidence, "confidence")
        if not 0.0 <= normalized_confidence <= 1.0:
            raise TimelineSchemaError("confidence must be between 0 and 1")
        if not isinstance(tool_active, bool):
            raise TimelineSchemaError("tool_active must be boolean")
        if tool_category is None:
            normalized_tool_category = ""
        elif not isinstance(tool_category, str):
            raise TimelineSchemaError("tool_category must be a string")
        else:
            normalized_tool_category = tool_category.strip().lower().replace("_", "-")
            if normalized_tool_category and not _SAFE_TOOL_CATEGORY_RE.match(normalized_tool_category):
                raise TimelineSchemaError("tool_category contains disallowed text")
        project_name = _project_name(project)
        session_hash = _stable_hash(session_id, self._salt)
        turn_hash = _stable_hash(turn_id, self._salt)
        safe_content = {
            "occurred_at": timestamp,
            "session_hash": session_hash,
            "turn_hash": turn_hash,
            "project": project_name,
            "kind": normalized_kind,
            "phase": normalized_phase,
            "source": normalized_source,
            "confidence": normalized_confidence,
            "tool_category": normalized_tool_category,
            "tool_active": bool(tool_active),
        }
        if event_id is None or (isinstance(event_id, str) and not event_id.strip()):
            event_id = hashlib.sha256(_canonical_safe_payload(safe_content).encode("utf-8")).hexdigest()
        else:
            event_id = _safe_id(event_id, "event_id")
        with self._lock:
            existing = self.connection.execute(
                "SELECT * FROM codex_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if existing is not None:
                existing_dict = self._event_row(existing)
                for key, value in safe_content.items():
                    if existing_dict[key] != value:
                        raise TimelineConflictError(
                            "event_id {} already exists with different immutable content".format(event_id)
                        )
                return event_id
            try:
                self.connection.execute(
                    "INSERT INTO codex_events(event_id,occurred_at,session_hash,turn_hash,project,"
                    "kind,phase,source,confidence,tool_category,tool_active,inserted_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        event_id,
                        timestamp,
                        session_hash,
                        turn_hash,
                        project_name,
                        normalized_kind,
                        normalized_phase,
                        normalized_source,
                        normalized_confidence,
                        normalized_tool_category,
                        int(tool_active),
                        time.time(),
                    ),
                )
            except sqlite3.Error as exc:
                raise TimelineError("cannot record Codex event: {}".format(exc)) from exc
        return event_id

    # Short aliases make the store convenient for collector/CLI code while
    # retaining the explicit contract spelling above.
    insert_codex_event = record_codex_event

    def record_gpu_sample(
        self,
        sampled_at: Any,
        host: Any = None,
        gpu_index: Optional[Any] = None,
        gpu_uuid_short: Any = "",
        state: Any = None,
        task_name: Any = "",
        attribution: Any = "",
        process_basename: Any = "",
        pid: Optional[Any] = None,
        sample_id: Optional[str] = None,
    ) -> str:
        """Append one GPU observation, retaining only safe display metadata."""

        # Collector adapters may hand a GPUSample-like object directly.  Keep
        # this convenience path attribute-based to avoid coupling the storage
        # module to the GPU probe implementation.
        if host is None and hasattr(sampled_at, "sampled_at"):
            observation = sampled_at
            sampled_at = getattr(observation, "sampled_at")
            host = getattr(observation, "host", None)
            gpu_index = getattr(observation, "gpu_index", None)
            gpu_uuid_short = getattr(observation, "gpu_uuid_short", "")
            state = getattr(observation, "state", None)
            task_name = getattr(observation, "task_name", task_name)
            attribution = getattr(observation, "attribution", attribution)
            process_basename = getattr(observation, "process_basename", process_basename)
            pid = getattr(observation, "pid", pid)
        timestamp = _timestamp(sampled_at, "sampled_at")
        if not isinstance(host, str):
            raise TimelineSchemaError("host must be a string")
        host = host.strip()
        if not _SAFE_HOST_RE.match(host):
            raise TimelineSchemaError("host contains disallowed text")
        if gpu_index is None:
            index = None
        else:
            try:
                index = int(gpu_index)
            except (TypeError, ValueError) as exc:
                raise TimelineSchemaError("gpu_index must be an integer or null") from exc
            if index < -1 or index > 255:
                raise TimelineSchemaError("gpu_index is outside the supported range")
        if gpu_uuid_short is None:
            uuid_short = ""
        else:
            uuid_short = _safe_label(gpu_uuid_short, "gpu_uuid_short", allow_empty=True)
        normalized_state = _normalize_gpu_state(state)
        normalized_task = _safe_label(task_name, "task_name", allow_empty=True)
        normalized_process = _basename(process_basename, "process_basename")
        if attribution is None:
            normalized_attribution = ""
        elif not isinstance(attribution, str):
            raise TimelineSchemaError("attribution must be a string")
        else:
            normalized_attribution = attribution.strip().lower().replace("_", "-")
            if normalized_attribution and not _SAFE_TOOL_CATEGORY_RE.match(normalized_attribution):
                raise TimelineSchemaError("attribution contains disallowed text")
        if pid is None:
            normalized_pid = None
        else:
            try:
                normalized_pid = int(pid)
            except (TypeError, ValueError) as exc:
                raise TimelineSchemaError("pid must be an integer or null") from exc
            if normalized_pid < 0:
                raise TimelineSchemaError("pid must be non-negative")
        safe_content = {
            "sampled_at": timestamp,
            "host": host,
            "gpu_index": index,
            "gpu_uuid_short": uuid_short,
            "state": normalized_state,
            "task_name": normalized_task,
            "attribution": normalized_attribution,
            "process_basename": normalized_process,
            "pid": normalized_pid,
        }
        if sample_id is None or (isinstance(sample_id, str) and not sample_id.strip()):
            sample_id = hashlib.sha256(_canonical_safe_payload(safe_content).encode("utf-8")).hexdigest()
        else:
            sample_id = _safe_id(sample_id, "sample_id")
        with self._lock:
            existing = self.connection.execute(
                "SELECT * FROM gpu_samples WHERE sample_id=?", (sample_id,)
            ).fetchone()
            if existing is not None:
                existing_dict = self._sample_row(existing)
                for key, value in safe_content.items():
                    if existing_dict[key] != value:
                        raise TimelineConflictError(
                            "sample_id {} already exists with different immutable content".format(sample_id)
                        )
                return sample_id
            try:
                self.connection.execute(
                    "INSERT INTO gpu_samples(sample_id,sampled_at,host,gpu_index,gpu_uuid_short,"
                    "state,task_name,attribution,process_basename,pid,inserted_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        sample_id,
                        timestamp,
                        host,
                        index,
                        uuid_short,
                        normalized_state,
                        normalized_task,
                        normalized_attribution,
                        normalized_process,
                        normalized_pid,
                        time.time(),
                    ),
                )
            except sqlite3.Error as exc:
                raise TimelineError("cannot record GPU sample: {}".format(exc)) from exc
        return sample_id

    insert_gpu_sample = record_gpu_sample

    def record_gpu_samples(self, samples: Any) -> None:
        """Append one probe pass atomically to avoid one fsync per GPU row."""

        rows = tuple(samples)
        if not rows:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                for sample in rows:
                    self.record_gpu_sample(sample)
                self.connection.execute("COMMIT")
            except Exception:
                try:
                    self.connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

    def record_override(
        self,
        target_type: Any,
        target_id: Any,
        field: Any,
        value: Any,
        override_id: Optional[str] = None,
        created_at: Optional[Any] = None,
        effective_from: Optional[Any] = None,
        effective_until: Optional[Any] = None,
        reason: Any = "",
    ) -> str:
        """Append an override without mutating its target observation."""

        normalized_target = _normalize_override_target(target_type)
        target_id = _safe_id(target_id, "target_id")
        if not isinstance(field, str):
            raise TimelineSchemaError("field must be a string")
        field = field.strip().lower().replace("-", "_")
        allowed_fields = {
            "codex_event": {"phase", "source", "confidence", "tool_category", "tool_active", "project"},
            "gpu_sample": {
                "state",
                "task_name",
                "attribution",
                "process_basename",
                "pid",
                "gpu_uuid_short",
            },
        }
        if field not in allowed_fields[normalized_target]:
            raise TimelineSchemaError("field {!r} cannot be overridden".format(field))
        # Validate override values using the same frozen vocabularies as raw
        # observations.  This avoids a report containing invented labels.
        if field == "phase":
            value = _normalize_phase(value)
        elif field == "state":
            value = _normalize_gpu_state(value)
        elif field == "source":
            value = _normalize_source(value)
        elif field == "confidence":
            value = _finite_number(value, "confidence")
            if not 0.0 <= value <= 1.0:
                raise TimelineSchemaError("confidence must be between 0 and 1")
        elif field == "tool_active":
            if not isinstance(value, bool):
                raise TimelineSchemaError("tool_active override must be boolean")
        elif field in ("task_name", "project", "gpu_uuid_short"):
            value = _project_name(value) if field == "project" else _safe_label(value, field, allow_empty=True)
        elif field == "process_basename":
            value = _basename(value, field)
        elif field == "pid":
            if value is not None:
                try:
                    value = int(value)
                except (TypeError, ValueError) as exc:
                    raise TimelineSchemaError("pid override must be an integer or null") from exc
                if value < 0:
                    raise TimelineSchemaError("pid override must be non-negative")
        encoded = _json_value(value, "value")
        now = time.time() if created_at is None else _timestamp(created_at, "created_at")
        start = now if effective_from is None else _timestamp(effective_from, "effective_from")
        until = None if effective_until is None else _timestamp(effective_until, "effective_until")
        if until is not None and until < start:
            raise TimelineSchemaError("effective_until must be >= effective_from")
        reason = _safe_label(reason, "reason", allow_empty=True)
        safe_content = {
            "target_type": normalized_target,
            "target_id": target_id,
            "field": field,
            "value": value,
            "created_at": now,
            "effective_from": start,
            "effective_until": until,
            "reason": reason,
        }
        if override_id is None or (isinstance(override_id, str) and not override_id.strip()):
            override_id = hashlib.sha256(_canonical_safe_payload(safe_content).encode("utf-8")).hexdigest()
        else:
            override_id = _safe_id(override_id, "override_id")
        with self._lock:
            existing = self.connection.execute(
                "SELECT * FROM timeline_overrides WHERE override_id=?", (override_id,)
            ).fetchone()
            if existing is not None:
                existing_dict = self._override_row(existing)
                for key, expected in safe_content.items():
                    if existing_dict[key] != expected:
                        raise TimelineConflictError(
                            "override_id {} already exists with different immutable content".format(override_id)
                        )
                return override_id
            try:
                self.connection.execute(
                    "INSERT INTO timeline_overrides(override_id,target_type,target_id,field,value_json,"
                    "created_at,effective_from,effective_until,reason) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        override_id,
                        normalized_target,
                        target_id,
                        field,
                        encoded,
                        now,
                        start,
                        until,
                        reason,
                    ),
                )
            except sqlite3.Error as exc:
                raise TimelineError("cannot record timeline override: {}".format(exc)) from exc
        return override_id

    insert_override = record_override
    add_override = record_override
    set_override = record_override

    def record_overrides(
        self,
        target_type: Any,
        target_id: Any,
        values: Mapping[str, Any],
        created_at: Optional[Any] = None,
        effective_from: Optional[Any] = None,
        effective_until: Optional[Any] = None,
        reason: Any = "",
    ) -> Tuple[str, ...]:
        """Append several scalar field overrides as independent records."""

        if not isinstance(values, Mapping) or not values:
            raise TimelineSchemaError("override values must be a non-empty object")
        result = []
        for field, value in values.items():
            result.append(
                self.record_override(
                    target_type,
                    target_id,
                    field,
                    value,
                    created_at=created_at,
                    effective_from=effective_from,
                    effective_until=effective_until,
                    reason=reason,
                )
            )
        return tuple(result)

    def list_codex_events(
        self,
        project: Optional[Any] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if project is not None:
            clauses.append("project=?")
            params.append(_project_name(project))
        if start is not None:
            clauses.append("occurred_at>=?")
            params.append(_timestamp(start, "start"))
        if end is not None:
            clauses.append("occurred_at<?")
            params.append(_timestamp(end, "end"))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            "SELECT * FROM codex_events" + where + " ORDER BY occurred_at, session_hash, event_id",
            tuple(params),
        ).fetchall()
        return [self._event_row(row) for row in rows]

    get_codex_events = list_codex_events
    iter_codex_events = list_codex_events
    list_events = list_codex_events

    def list_gpu_samples(
        self,
        host: Optional[Any] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if host is not None:
            if not isinstance(host, str) or not _SAFE_HOST_RE.match(host.strip()):
                raise TimelineSchemaError("host contains disallowed text")
            clauses.append("host=?")
            params.append(host.strip())
        if start is not None:
            clauses.append("sampled_at>=?")
            params.append(_timestamp(start, "start"))
        if end is not None:
            clauses.append("sampled_at<?")
            params.append(_timestamp(end, "end"))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            "SELECT * FROM gpu_samples" + where + " ORDER BY sampled_at, host, gpu_index, sample_id",
            tuple(params),
        ).fetchall()
        return [self._sample_row(row) for row in rows]

    get_gpu_samples = list_gpu_samples
    iter_gpu_samples = list_gpu_samples
    list_samples = list_gpu_samples

    def list_overrides(
        self,
        target_type: Optional[Any] = None,
        target_id: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if target_type is not None:
            clauses.append("target_type=?")
            params.append(_normalize_override_target(target_type))
        if target_id is not None:
            clauses.append("target_id=?")
            params.append(_safe_id(target_id, "target_id"))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            "SELECT * FROM timeline_overrides" + where + " ORDER BY effective_from, created_at, override_id",
            tuple(params),
        ).fetchall()
        return [self._override_row(row) for row in rows]

    get_overrides = list_overrides

    def current_overrides(
        self,
        target_type: Any,
        target_id: Any,
        at: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return the latest valid field overrides for one target."""

        target_type = _normalize_override_target(target_type)
        target_id = _safe_id(target_id, "target_id")
        when = time.time() if at is None else _timestamp(at, "at")
        rows = self.connection.execute(
            "SELECT * FROM timeline_overrides WHERE target_type=? AND target_id=? "
            "AND effective_from<=? AND (effective_until IS NULL OR effective_until>=?) "
            "ORDER BY effective_from DESC, created_at DESC, override_id DESC",
            (target_type, target_id, when, when),
        ).fetchall()
        result: Dict[str, Any] = {}
        for row in rows:
            item = self._override_row(row)
            result.setdefault(item["field"], item["value"])
        return result

    resolve_overrides = current_overrides

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        """Read-only diagnostic query helper; mutation is intentionally blocked."""

        if not isinstance(sql, str) or not sql.lstrip().lower().startswith("select"):
            raise TimelineSchemaError("query only accepts SELECT statements")
        return self.connection.execute(sql, tuple(params)).fetchall()


# Backwards/short names used by integrations and tests.
Store = TimelineStore
TimelineDB = TimelineStore


__all__ = [
    "DEFAULT_DB_PATH",
    "EVENT_SOURCES",
    "GPU_STATES",
    "SCHEMA_VERSION",
    "Store",
    "TimelineConflictError",
    "TimelineDB",
    "TimelineError",
    "TimelineSchemaError",
    "TimelineStore",
    "default_db_path",
]
