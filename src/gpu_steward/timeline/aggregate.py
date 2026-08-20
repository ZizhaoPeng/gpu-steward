"""Deterministic daily aggregation for the local timeline database.

The aggregator converts point observations into half-open intervals.  It is
deliberately independent of the queue scheduler: queue state remains the
Control Plane, while this module is the read-only Observe Plane.
"""

from __future__ import annotations

import datetime as _datetime
import math
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .phases import CODEX_PHASES
from .store import (
    GPU_STATES,
    TimelineSchemaError,
    TimelineStore,
    _normalize_gpu_state,
    _project_name,
)


SCHEMA_VERSION = 1
DEFAULT_TIMEZONE = "Asia/Singapore"
STALL_SECONDS = 10 * 60
DEFAULT_SAMPLE_GAP_SECONDS = 10 * 60
DEFAULT_DISPLAY_MERGE_GAP_SECONDS = 10 * 60

_WAITING_PHASES = frozenset(("waiting-tool", "waiting-user"))
_ACTIVE_PHASES = frozenset(
    phase
    for phase in CODEX_PHASES
    if phase not in _WAITING_PHASES and phase not in ("suspected-stall", "idle")
)


def _as_timestamp(value: Any, field: str) -> float:
    if isinstance(value, _datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_datetime.timezone.utc)
        result = value.timestamp()
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TimelineSchemaError("{} must be a timestamp".format(field)) from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise TimelineSchemaError("{} must be finite".format(field))
    return result


def _timezone(name: str) -> _datetime.tzinfo:
    if not isinstance(name, str) or not name.strip():
        raise TimelineSchemaError("timezone must be a non-empty string")
    name = name.strip()
    # The frozen product timezone is a fixed UTC+08 display boundary.  Using
    # the historical IANA offset for synthetic pre-1982 dates (Singapore had
    # a short-lived +07:30 offset) would make reports and tests depend on data
    # vintage rather than the product contract.
    if name in ("Asia/Singapore", "Asia/Shanghai"):
        return _datetime.timezone(_datetime.timedelta(hours=8), name)
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+

        return ZoneInfo(name)
    except (ImportError, KeyError):
        # Python 3.8 has no zoneinfo in the standard library.  Singapore is a
        # fixed UTC+08 timezone, as are these common aliases; fail closed for
        # arbitrary DST zones rather than silently using the wrong boundary.
        fixed = {
            "UTC": _datetime.timezone.utc,
            "Etc/UTC": _datetime.timezone.utc,
            "Asia/Singapore": _datetime.timezone(_datetime.timedelta(hours=8), "Asia/Singapore"),
            "Asia/Shanghai": _datetime.timezone(_datetime.timedelta(hours=8), "Asia/Shanghai"),
        }
        if name in fixed:
            return fixed[name]
        raise TimelineSchemaError("timezone {!r} is unavailable".format(name))


def _day_bounds(date_value: Any, timezone: str) -> Tuple[str, float, float, _datetime.tzinfo]:
    tz = _timezone(timezone)
    if isinstance(date_value, _datetime.datetime):
        local_date = date_value.astimezone(tz).date() if date_value.tzinfo else date_value.date()
    elif isinstance(date_value, _datetime.date):
        local_date = date_value
    elif isinstance(date_value, str):
        try:
            local_date = _datetime.date.fromisoformat(date_value.strip())
        except (TypeError, ValueError) as exc:
            raise TimelineSchemaError("date must be YYYY-MM-DD") from exc
        if local_date.isoformat() != date_value.strip():
            raise TimelineSchemaError("date must be YYYY-MM-DD")
    else:
        raise TimelineSchemaError("date must be YYYY-MM-DD")
    start_local = _datetime.datetime.combine(local_date, _datetime.time.min).replace(tzinfo=tz)
    end_local = start_local + _datetime.timedelta(days=1)
    return local_date.isoformat(), start_local.timestamp(), end_local.timestamp(), tz


def _clip(start: float, end: float, window_start: float, window_end: float) -> Optional[Tuple[float, float]]:
    left = max(float(start), window_start)
    right = min(float(end), window_end)
    if right <= left:
        return None
    return left, right


def _metadata_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = set(left) | set(right)
    return all(left.get(key) == right.get(key) for key in keys if key not in ("start", "end", "duration_seconds"))


def _append_segment(segments: List[Dict[str, Any]], start: float, end: float, metadata: Mapping[str, Any]) -> None:
    if end <= start:
        return
    item = dict(metadata)
    item["start"] = float(start)
    item["end"] = float(end)
    item["duration_seconds"] = float(end - start)
    if segments and abs(segments[-1]["end"] - start) < 1e-9 and _metadata_equal(segments[-1], item):
        segments[-1]["end"] = float(end)
        segments[-1]["duration_seconds"] = float(segments[-1]["end"] - segments[-1]["start"])
        return
    segments.append(item)


def _gpu_task_family(segment: Mapping[str, Any]) -> str:
    """Return a stable, privacy-bounded display identity for one GPU task."""

    if str(segment.get("state")) in ("idle", "unknown"):
        return ""
    task_name = str(segment.get("task_name") or "").strip()
    process_name = str(segment.get("process_basename") or "").strip()
    if segment.get("attribution") == "explicit" and task_name:
        return task_name
    inferred = (process_name or task_name).lower()
    if inferred.startswith("python"):
        return "Python 训练/计算进程"
    if inferred in ("", "no data", "n/a", "unknown"):
        return "外部 GPU 进程"
    return process_name or task_name


def _gpu_display_key(segment: Mapping[str, Any]) -> Tuple[str, str]:
    return str(segment.get("state") or "unknown"), _gpu_task_family(segment)


def _occupied_gpu_segment(segment: Mapping[str, Any]) -> bool:
    return str(segment.get("state")) in {
        "training", "managed-other", "external", "reserved", "disabled"
    } and bool(_gpu_task_family(segment))


def _merge_gpu_pair(left: Dict[str, Any], right: Mapping[str, Any], gap_seconds: float = 0.0) -> None:
    left["end"] = float(right["end"])
    left["duration_seconds"] = float(left["end"] - left["start"])
    left["observed_seconds"] = float(
        left.get("observed_seconds", 0.0) + right.get("observed_seconds", right["duration_seconds"])
    )
    left["gap_seconds"] = float(left.get("gap_seconds", 0.0) + gap_seconds + right.get("gap_seconds", 0.0))
    left["gap_count"] = int(left.get("gap_count", 0) + (1 if gap_seconds > 0 else 0) + right.get("gap_count", 0))
    left["sample_count"] = int(left.get("sample_count", 1) + right.get("sample_count", 1))
    left["last_sample_id"] = right.get("last_sample_id", right.get("sample_id"))


def _coalesce_gpu_segments(
    segments: Sequence[Mapping[str, Any]], merge_gap_seconds: float
) -> List[Dict[str, Any]]:
    """Build readable display bars without changing raw samples or summary totals."""

    prepared: List[Dict[str, Any]] = []
    for raw in segments:
        item = dict(raw)
        item["task_name"] = _gpu_task_family(item)
        item["observed_seconds"] = float(item["duration_seconds"])
        item["gap_seconds"] = 0.0
        item["gap_count"] = 0
        item["sample_count"] = 1
        item["first_sample_id"] = item.get("sample_id")
        item["last_sample_id"] = item.get("sample_id")
        if prepared and abs(prepared[-1]["end"] - item["start"]) < 1e-9:
            if _gpu_display_key(prepared[-1]) == _gpu_display_key(item):
                _merge_gpu_pair(prepared[-1], item)
                continue
        prepared.append(item)

    if merge_gap_seconds <= 0:
        return prepared
    result: List[Dict[str, Any]] = []
    index = 0
    while index < len(prepared):
        current = dict(prepared[index])
        if _occupied_gpu_segment(current):
            gap_end = index + 1
            gap_total = 0.0
            while gap_end < len(prepared) and prepared[gap_end].get("state") in ("idle", "unknown"):
                gap_total += float(prepared[gap_end]["duration_seconds"])
                gap_end += 1
            if (
                gap_end > index + 1
                and gap_end < len(prepared)
                and gap_total <= merge_gap_seconds
                and _gpu_display_key(current) == _gpu_display_key(prepared[gap_end])
            ):
                _merge_gpu_pair(current, prepared[gap_end], gap_seconds=gap_total)
                prepared[gap_end] = current
                index = gap_end
                continue
        result.append(current)
        index += 1
    return result


def _event_with_overrides(store: TimelineStore, row: Mapping[str, Any], at: float) -> Dict[str, Any]:
    value = dict(row)
    value.update(store.current_overrides("codex_event", row["event_id"], at=at))
    return value


def _sample_with_overrides(store: TimelineStore, row: Mapping[str, Any], at: float) -> Dict[str, Any]:
    value = dict(row)
    value.update(store.current_overrides("gpu_sample", row["sample_id"], at=at))
    return value


def _event_state(row: Mapping[str, Any], previous_phase: Optional[str]) -> Tuple[str, str, bool]:
    kind = row["kind"]
    phase = row["phase"]
    tool_active = bool(row.get("tool_active", False))
    if kind == "pre-tool":
        return "waiting-tool", "waiting-tool", True
    if kind in ("stop", "turn-end"):
        return "waiting-user", "waiting-user", False
    if kind == "post-tool":
        # PostToolUse closes waiting-tool.  If no explicit semantic phase was
        # supplied, restore the last active phase instead of losing research /
        # review/etc. to the generic active-unspecified label.
        if phase in _WAITING_PHASES or phase in ("suspected-stall", "idle"):
            phase = previous_phase if previous_phase in _ACTIVE_PHASES else "active-unspecified"
        return "active", phase, False
    if phase == "waiting-tool":
        return "waiting-tool", phase, tool_active
    if phase == "waiting-user":
        return "waiting-user", phase, False
    if phase == "suspected-stall":
        return "stalled", phase, False
    if phase == "idle":
        return "idle", phase, False
    if phase not in _ACTIVE_PHASES:
        raise TimelineSchemaError("unknown Codex phase in stored event: {!r}".format(phase))
    return "active", phase, tool_active


def _codex_lanes(
    store: TimelineStore,
    window_start: float,
    window_end: float,
    as_of: float,
    project: Optional[Any],
    stall_seconds: float,
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]], Dict[str, float]]:
    project_name = None if project is None else _project_name(project)
    events = store.list_codex_events(project=project_name)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw in events:
        grouped[raw["session_hash"]].append(_event_with_overrides(store, raw, as_of))

    lanes: List[Dict[str, Any]] = []
    active_intervals: List[Tuple[float, float]] = []
    totals = {
        "codex_active_seconds": 0.0,
        "codex_waiting_seconds": 0.0,
        "codex_stalled_seconds": 0.0,
    }
    for session_hash in sorted(grouped):
        rows = sorted(grouped[session_hash], key=lambda row: (row["occurred_at"], row["event_id"]))
        segments: List[Dict[str, Any]] = []
        previous_at: Optional[float] = None
        state: Optional[str] = None
        current_phase: Optional[str] = None
        current_tool_active = False
        current_meta: Dict[str, Any] = {}
        for row in rows:
            occurred_at = float(row["occurred_at"])
            if previous_at is not None and state is not None and occurred_at > previous_at:
                _append_codex_interval(
                    segments,
                    previous_at,
                    occurred_at,
                    state,
                    current_phase or "active-unspecified",
                    current_meta,
                    window_start,
                    window_end,
                    totals,
                    active_intervals,
                )
            state, current_phase, current_tool_active = _event_state(row, current_phase)
            current_meta = {
                "phase": current_phase,
                "source": row["source"],
                "confidence": float(row["confidence"]),
                "tool_category": row.get("tool_category", ""),
                "tool_active": bool(current_tool_active),
                "event_id": row["event_id"],
            }
            previous_at = occurred_at
        if previous_at is None or state is None:
            continue
        final_end = min(window_end, as_of)
        if final_end > previous_at:
            # Only an unfinished, non-tool active phase can become a derived
            # stall.  A long-running PreToolUse interval remains waiting-tool
            # for as long as the hook has not delivered PostToolUse.
            stall_at = previous_at + float(stall_seconds)
            if state == "active" and not current_tool_active and final_end > stall_at:
                _append_codex_interval(
                    segments,
                    previous_at,
                    stall_at,
                    state,
                    current_phase or "active-unspecified",
                    current_meta,
                    window_start,
                    window_end,
                    totals,
                    active_intervals,
                )
                _append_codex_interval(
                    segments,
                    stall_at,
                    final_end,
                    "stalled",
                    "suspected-stall",
                    {
                        "phase": "suspected-stall",
                        "source": "inferred",
                        "confidence": 0.5,
                        "tool_category": "",
                        "tool_active": False,
                        "event_id": "stall:{}:{}".format(session_hash, int(stall_at)),
                    },
                    window_start,
                    window_end,
                    totals,
                    active_intervals,
                )
            else:
                _append_codex_interval(
                    segments,
                    previous_at,
                    final_end,
                    state,
                    current_phase or "active-unspecified",
                    current_meta,
                    window_start,
                    window_end,
                    totals,
                    active_intervals,
                )
        if segments:
            lanes.append(
                {
                    "id": "codex:{}".format(session_hash),
                    "kind": "codex",
                    "label": "Codex",
                    "segments": segments,
                }
            )
    return lanes, active_intervals, totals


def _append_codex_interval(
    segments: List[Dict[str, Any]],
    start: float,
    end: float,
    state: str,
    phase: str,
    metadata: Mapping[str, Any],
    window_start: float,
    window_end: float,
    totals: Dict[str, float],
    active_intervals: List[Tuple[float, float]],
) -> None:
    clipped = _clip(start, end, window_start, window_end)
    if clipped is None:
        return
    left, right = clipped
    item = {
        "phase": phase,
        "source": metadata.get("source", "hook-rule"),
        "confidence": float(metadata.get("confidence", 0.8)),
        "tool_category": metadata.get("tool_category", ""),
        "tool_active": bool(metadata.get("tool_active", False)),
    }
    if "event_id" in metadata:
        item["event_id"] = metadata["event_id"]
    _append_segment(segments, left, right, item)
    duration = right - left
    if state == "active":
        totals["codex_active_seconds"] += duration
        active_intervals.append((left, right))
    elif state in ("waiting", "waiting-tool", "waiting-user"):
        totals["codex_waiting_seconds"] += duration
    elif state == "stalled":
        totals["codex_stalled_seconds"] += duration


def _gpu_lanes(
    store: TimelineStore,
    window_start: float,
    window_end: float,
    as_of: float,
    sample_gap_seconds: Optional[float],
    display_merge_gap_seconds: float,
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]], Dict[str, float]]:
    samples = store.list_gpu_samples()
    grouped: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = defaultdict(list)
    for raw in samples:
        grouped[(raw["host"], raw["gpu_index"])].append(_sample_with_overrides(store, raw, as_of))
    lanes: List[Dict[str, Any]] = []
    training_intervals: List[Tuple[float, float]] = []
    totals = {"gpu_training_seconds": 0.0, "gpu_idle_seconds": 0.0}
    for (host, gpu_index) in sorted(grouped, key=lambda item: (item[0], -1 if item[1] is None else item[1])):
        rows = sorted(grouped[(host, gpu_index)], key=lambda row: (row["sampled_at"], row["sample_id"]))
        segments: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            start = float(row["sampled_at"])
            if index + 1 < len(rows):
                next_at = float(rows[index + 1]["sampled_at"])
                if next_at <= start:
                    continue
                end = min(next_at, as_of, window_end)
                state_end = end
                gap = None if sample_gap_seconds is None else float(sample_gap_seconds)
                if gap is not None and gap > 0 and next_at - start > gap:
                    state_end = min(start + gap, end)
                _append_gpu_interval(
                    segments,
                    start,
                    state_end,
                    row,
                    window_start,
                    window_end,
                    totals,
                    training_intervals,
                )
                if gap is not None and gap > 0 and next_at - start > gap:
                    _append_gpu_interval(
                        segments,
                        start + gap,
                        end,
                        dict(row, state="unknown", task_name="", attribution="inferred", process_basename="", pid=None),
                        window_start,
                        window_end,
                        totals,
                        training_intervals,
                    )
            else:
                final_end = min(as_of, window_end)
                state_end = final_end
                gap = None if sample_gap_seconds is None else float(sample_gap_seconds)
                stale_after = start + gap if gap is not None and gap > 0 else None
                _append_gpu_interval(
                    segments,
                    start,
                    min(state_end, stale_after) if stale_after is not None else state_end,
                    row,
                    window_start,
                    window_end,
                    totals,
                    training_intervals,
                )
                if stale_after is not None and final_end > stale_after:
                    _append_gpu_interval(
                        segments,
                        stale_after,
                        final_end,
                        dict(row, state="unknown", task_name="", attribution="inferred", process_basename="", pid=None),
                        window_start,
                        window_end,
                        totals,
                        training_intervals,
                    )
        if segments:
            segments = _coalesce_gpu_segments(segments, display_merge_gap_seconds)
            suffix = "unknown" if gpu_index is None else str(gpu_index)
            label_index = "unknown" if gpu_index is None else str(gpu_index)
            lanes.append(
                {
                    "id": "gpu:{}:{}".format(host, suffix),
                    "kind": "gpu",
                    "label": "{} / GPU {}".format(host, label_index),
                    "segments": segments,
                }
            )
    return lanes, training_intervals, totals


def _append_gpu_interval(
    segments: List[Dict[str, Any]],
    start: float,
    end: float,
    row: Mapping[str, Any],
    window_start: float,
    window_end: float,
    totals: Dict[str, float],
    training_intervals: List[Tuple[float, float]],
) -> None:
    clipped = _clip(start, end, window_start, window_end)
    if clipped is None:
        return
    left, right = clipped
    try:
        state = _normalize_gpu_state(row["state"])
    except TimelineSchemaError:
        # The store API and schema are already defensive, but keep aggregation
        # fail-closed if an operator imported a malformed row directly.
        raise
    item = {
        "state": state,
        "task_name": row.get("task_name", ""),
        "attribution": row.get("attribution", ""),
        "process_basename": row.get("process_basename", ""),
        "pid": row.get("pid"),
        "sample_id": row.get("sample_id"),
    }
    _append_segment(segments, left, right, item)
    duration = right - left
    if state == "training":
        totals["gpu_training_seconds"] += duration
        training_intervals.append((left, right))
    elif state == "idle":
        totals["gpu_idle_seconds"] += duration


def _union(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted((float(left), float(right)) for left, right in intervals if right > left)
    result: List[Tuple[float, float]] = []
    for left, right in ordered:
        if not result or left > result[-1][1]:
            result.append((left, right))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], right))
    return result


def _intersection_duration(left: Iterable[Tuple[float, float]], right: Iterable[Tuple[float, float]]) -> float:
    first = _union(left)
    second = _union(right)
    index_a = index_b = 0
    total = 0.0
    while index_a < len(first) and index_b < len(second):
        start = max(first[index_a][0], second[index_b][0])
        end = min(first[index_a][1], second[index_b][1])
        if end > start:
            total += end - start
        if first[index_a][1] < second[index_b][1]:
            index_a += 1
        else:
            index_b += 1
    return total


def _round_number(value: float) -> float:
    # Keep JSON readable while retaining sub-second observations.  A whole
    # second is represented as an int-like float, which compares naturally in
    # tests and does not lose data in the store.
    rounded = round(float(value), 6)
    return rounded


class TimelineAggregator:
    """Read-only report builder bound to one :class:`TimelineStore`."""

    def __init__(self, store: TimelineStore):
        if not isinstance(store, TimelineStore):
            raise TimelineSchemaError("store must be a TimelineStore")
        self.store = store

    def build_day(
        self,
        date: Any,
        timezone: str = DEFAULT_TIMEZONE,
        project: Optional[Any] = None,
        generated_at: Optional[Any] = None,
        stall_seconds: float = STALL_SECONDS,
        sample_gap_seconds: Optional[float] = DEFAULT_SAMPLE_GAP_SECONDS,
        display_merge_gap_seconds: float = DEFAULT_DISPLAY_MERGE_GAP_SECONDS,
    ) -> Dict[str, Any]:
        date_text, day_start, day_end, _tz = _day_bounds(date, timezone)
        as_of = time.time() if generated_at is None else _as_timestamp(generated_at, "generated_at")
        stall_seconds = _as_timestamp(stall_seconds, "stall_seconds")
        if stall_seconds <= 0:
            raise TimelineSchemaError("stall_seconds must be positive")
        if sample_gap_seconds is not None:
            sample_gap_seconds = _as_timestamp(sample_gap_seconds, "sample_gap_seconds")
            if sample_gap_seconds <= 0:
                raise TimelineSchemaError("sample_gap_seconds must be positive or null")
        display_merge_gap_seconds = _as_timestamp(
            display_merge_gap_seconds, "display_merge_gap_seconds"
        )
        if display_merge_gap_seconds < 0:
            raise TimelineSchemaError("display_merge_gap_seconds must be non-negative")

        effective_end = min(day_end, as_of)
        lanes: List[Dict[str, Any]] = []
        active_intervals: List[Tuple[float, float]] = []
        training_intervals: List[Tuple[float, float]] = []
        totals = {
            "codex_active_seconds": 0.0,
            "codex_waiting_seconds": 0.0,
            "codex_stalled_seconds": 0.0,
            "gpu_training_seconds": 0.0,
            "gpu_idle_seconds": 0.0,
        }
        if effective_end > day_start:
            codex, active_intervals, codex_totals = _codex_lanes(
                self.store,
                day_start,
                effective_end,
                as_of,
                project,
                stall_seconds,
            )
            gpu, training_intervals, gpu_totals = _gpu_lanes(
                self.store,
                day_start,
                effective_end,
                as_of,
                sample_gap_seconds,
                display_merge_gap_seconds,
            )
            lanes.extend(codex)
            lanes.extend(gpu)
            totals.update(codex_totals)
            totals.update(gpu_totals)

        summary = {key: _round_number(value) for key, value in totals.items()}
        summary["overlap_seconds"] = _round_number(_intersection_duration(active_intervals, training_intervals))
        return {
            "schema_version": SCHEMA_VERSION,
            "date": date_text,
            "timezone": timezone,
            "generated_at": _round_number(as_of),
            "display_merge_gap_seconds": _round_number(display_merge_gap_seconds),
            "lanes": lanes,
            "summary": summary,
        }


def build_day(
    store: TimelineStore,
    date: Any,
    timezone: str = DEFAULT_TIMEZONE,
    project: Optional[Any] = None,
    generated_at: Optional[Any] = None,
    stall_seconds: float = STALL_SECONDS,
    sample_gap_seconds: Optional[float] = DEFAULT_SAMPLE_GAP_SECONDS,
    display_merge_gap_seconds: float = DEFAULT_DISPLAY_MERGE_GAP_SECONDS,
) -> Dict[str, Any]:
    """Build one local-day report from a private timeline store."""

    return TimelineAggregator(store).build_day(
        date,
        timezone=timezone,
        project=project,
        generated_at=generated_at,
        stall_seconds=stall_seconds,
        sample_gap_seconds=sample_gap_seconds,
        display_merge_gap_seconds=display_merge_gap_seconds,
    )


def derive_stalls(
    store: TimelineStore,
    as_of: Optional[Any] = None,
    project: Optional[Any] = None,
    stall_seconds: float = STALL_SECONDS,
) -> List[Dict[str, Any]]:
    """Return derived suspected-stall intervals without writing an event."""

    now = time.time() if as_of is None else _as_timestamp(as_of, "as_of")
    # A broad UTC window is enough for this diagnostic API; build_day performs
    # the authoritative local-day clipping.
    lanes, _active, _totals = _codex_lanes(
        store,
        -float("inf"),
        now,
        now,
        project,
        _as_timestamp(stall_seconds, "stall_seconds"),
    )
    result: List[Dict[str, Any]] = []
    for lane in lanes:
        for segment in lane["segments"]:
            if segment["phase"] == "suspected-stall":
                result.append(dict(segment, lane_id=lane["id"]))
    return result


aggregate_day = build_day
find_stalls = derive_stalls


__all__ = [
    "DEFAULT_SAMPLE_GAP_SECONDS",
    "DEFAULT_DISPLAY_MERGE_GAP_SECONDS",
    "DEFAULT_TIMEZONE",
    "SCHEMA_VERSION",
    "STALL_SECONDS",
    "TimelineAggregator",
    "aggregate_day",
    "build_day",
    "derive_stalls",
    "find_stalls",
]
