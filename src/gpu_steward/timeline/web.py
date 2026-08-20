"""Local-only HTTP dashboard for the GPU Steward Observe Plane.

The web layer deliberately knows very little about collection or aggregation.  A
report builder is injected by the caller (the production CLI will provide the
timeline aggregate builder, while tests can use a small deterministic fixture).
This keeps the dashboard read-only and keeps the queue database separate from
the timeline database.
"""

from __future__ import annotations

import json
from datetime import date as date_type
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qs, urlsplit


LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TIMEZONE = "Asia/Singapore"
REPORT_SCHEMA_VERSION = 1
ASSET_ROOT = Path(__file__).resolve().parent / "assets"

_SUMMARY_KEYS = (
    "codex_active_seconds",
    "codex_waiting_seconds",
    "codex_stalled_seconds",
    "gpu_training_seconds",
    "gpu_idle_seconds",
    "overlap_seconds",
)
_LANE_KINDS = frozenset(("codex", "gpu"))
_REPORT_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
}

ReportBuilder = Callable[..., Mapping[str, Any]]


class TimelineWebError(ValueError):
    """Raised when a report or web request violates the frozen web contract."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_date(value: str) -> str:
    candidate = str(value or "").strip()
    try:
        parsed = date_type.fromisoformat(candidate)
    except (TypeError, ValueError) as exc:
        raise TimelineWebError("date must use YYYY-MM-DD") from exc
    if parsed.isoformat() != candidate:
        raise TimelineWebError("date must use YYYY-MM-DD")
    return candidate


def _default_report_date(timezone: str) -> str:
    # The MVP timezone is deliberately fixed.  Avoid depending on zoneinfo so
    # the Python 3.8 standard-library runtime remains supported.
    if timezone == DEFAULT_TIMEZONE:
        return (datetime.utcnow() + timedelta(hours=8)).date().isoformat()
    return datetime.utcnow().date().isoformat()


def _default_report_builder(
    *, date: str, timezone: str, project: Optional[str]
) -> Mapping[str, Any]:
    """Load the aggregate builder lazily to keep the web module importable.

    The timeline core is implemented independently of this worker.  Delayed
    import means the dashboard tests can inject a fake report and do not need a
    writable home directory or a live GPU collector.
    """

    try:
        from .aggregate import build_day  # type: ignore
        from .store import TimelineStore  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by CLI smoke only
        raise TimelineWebError("timeline aggregate builder is unavailable") from exc
    store = TimelineStore()
    try:
        return build_day(store, date, timezone=timezone, project=project)
    finally:
        store.close()


def validate_report(
    report: Mapping[str, Any], *, requested_date: str, timezone: str
) -> Dict[str, Any]:
    """Validate the public report envelope before it crosses the HTTP boundary."""

    if not isinstance(report, Mapping):
        raise TimelineWebError("report builder must return an object")
    payload = dict(report)
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise TimelineWebError("unsupported timeline report schema version")
    if payload.get("date") != requested_date:
        raise TimelineWebError("report date does not match the request")
    if payload.get("timezone") != timezone:
        raise TimelineWebError("report timezone does not match the dashboard")
    if not _is_number(payload.get("generated_at")):
        raise TimelineWebError("report generated_at must be numeric")

    lanes = payload.get("lanes")
    if not isinstance(lanes, list):
        raise TimelineWebError("report lanes must be an array")
    for lane in lanes:
        if not isinstance(lane, Mapping):
            raise TimelineWebError("each report lane must be an object")
        if not isinstance(lane.get("id"), str) or not lane.get("id"):
            raise TimelineWebError("each report lane needs a non-empty id")
        if lane.get("kind") not in _LANE_KINDS:
            raise TimelineWebError("each report lane kind must be codex or gpu")
        if not isinstance(lane.get("label"), str) or not lane.get("label"):
            raise TimelineWebError("each report lane needs a non-empty label")
        segments = lane.get("segments")
        if not isinstance(segments, list):
            raise TimelineWebError("each report lane segments value must be an array")
        for segment in segments:
            if not isinstance(segment, Mapping):
                raise TimelineWebError("each timeline segment must be an object")
            start = segment.get("start")
            end = segment.get("end")
            if not _is_number(start) or not _is_number(end) or end <= start:
                raise TimelineWebError("segment start/end must be increasing numbers")
            source = segment.get("source")
            if source is not None and source not in (
                "declared",
                "hook-rule",
                "inferred",
            ):
                raise TimelineWebError("unknown segment source")
            confidence = segment.get("confidence")
            if confidence is not None and (
                not _is_number(confidence) or confidence < 0 or confidence > 1
            ):
                raise TimelineWebError("segment confidence must be between 0 and 1")
            for key in ("duration_seconds", "observed_seconds", "gap_seconds"):
                value = segment.get(key)
                if value is not None and (not _is_number(value) or value < 0):
                    raise TimelineWebError("segment durations must be non-negative numbers")
            gap_count = segment.get("gap_count")
            if gap_count is not None and (
                not isinstance(gap_count, int) or isinstance(gap_count, bool) or gap_count < 0
            ):
                raise TimelineWebError("segment gap_count must be a non-negative integer")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise TimelineWebError("report summary must be an object")
    for key in _SUMMARY_KEYS:
        value = summary.get(key)
        if not _is_number(value) or value < 0:
            raise TimelineWebError("summary values must be non-negative numbers")
    return payload


class TimelineWebApp:
    """Read-only application state shared by HTTP handler instances."""

    def __init__(
        self,
        report_builder: Optional[ReportBuilder] = None,
        *,
        timezone: str = DEFAULT_TIMEZONE,
        project: Optional[str] = None,
    ) -> None:
        if not timezone or not isinstance(timezone, str):
            raise ValueError("timezone must be a non-empty string")
        if project is not None and len(project) > 256:
            raise ValueError("project filter is too long")
        self.report_builder = report_builder or _default_report_builder
        self.timezone = timezone
        self.project = project

    def report(self, requested_date: str, project: Optional[str] = None) -> Dict[str, Any]:
        normalized_date = _validate_date(requested_date)
        selected_project = self.project if project is None else project
        if selected_project is not None and len(selected_project) > 256:
            raise TimelineWebError("project filter is too long")
        report = self.report_builder(
            date=normalized_date,
            timezone=self.timezone,
            project=selected_project,
        )
        return validate_report(
            report,
            requested_date=normalized_date,
            timezone=self.timezone,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "ok": True,
            "service": "gpu-steward-timeline",
            "plane": "observe",
            "timezone": self.timezone,
        }


class TimelineHTTPServer(ThreadingHTTPServer):
    """Threaded localhost server carrying an immutable application reference."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], app: TimelineWebApp) -> None:
        self.app = app
        super().__init__(server_address, TimelineRequestHandler)


class TimelineRequestHandler(BaseHTTPRequestHandler):
    """Serve static local assets and the validated report API."""

    server: TimelineHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        if parsed.path == "/favicon.ico" and not parsed.query:
            # Browsers request this implicitly.  A no-content response keeps
            # the local smoke test and console clean without adding an asset.
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if parsed.path == "/api/health":
            self._send_json(HTTPStatus.OK, self.server.app.health())
            return
        if parsed.path == "/api/report":
            self._handle_report(parsed.query)
            return
        asset = _REPORT_ASSETS.get(parsed.path)
        if asset is None or parsed.query:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        filename, content_type = asset
        try:
            content = (ASSET_ROOT / filename).read_bytes()
        except OSError:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "asset unavailable"})
            return
        self._send_bytes(HTTPStatus.OK, content_type, content)

    def _handle_report(self, query: str) -> None:
        values = parse_qs(query, keep_blank_values=True, strict_parsing=False)
        raw_date = values.get("date", [None])[0]
        requested_date = raw_date or _default_report_date(self.server.app.timezone)
        project = values.get("project", [None])[0]
        try:
            report = self.server.app.report(requested_date, project)
        except TimelineWebError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        except Exception:
            # Collection/DB failures are reported without exposing paths,
            # command arguments, credentials, or training logs.
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": "report unavailable"},
            )
            return
        self._send_json(HTTPStatus.OK, report)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", body)

    def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Dashboard access is local-only and must not become a telemetry sink.
        return


def create_server(
    report_builder: Optional[ReportBuilder] = None,
    *,
    host: str = LOCALHOST,
    port: int = DEFAULT_PORT,
    timezone: str = DEFAULT_TIMEZONE,
    project: Optional[str] = None,
) -> TimelineHTTPServer:
    """Create, but do not start, a localhost-only dashboard server."""

    if host != LOCALHOST:
        raise ValueError("GPU Steward Timeline dashboard only binds to 127.0.0.1")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    return TimelineHTTPServer(
        (LOCALHOST, port),
        TimelineWebApp(report_builder, timezone=timezone, project=project),
    )


def serve(
    report_builder: Optional[ReportBuilder] = None,
    *,
    host: str = LOCALHOST,
    port: int = DEFAULT_PORT,
    timezone: str = DEFAULT_TIMEZONE,
    project: Optional[str] = None,
) -> None:
    """Run the dashboard until interrupted, always closing its socket."""

    server = create_server(
        report_builder,
        host=host,
        port=port,
        timezone=timezone,
        project=project,
    )
    try:
        # The dashboard has no automatic refresh, so a relaxed accept-loop
        # cadence keeps an open but inactive page effectively idle.
        server.serve_forever(poll_interval=2.0)
    finally:
        server.server_close()


__all__ = [
    "ASSET_ROOT",
    "DEFAULT_PORT",
    "DEFAULT_TIMEZONE",
    "LOCALHOST",
    "REPORT_SCHEMA_VERSION",
    "TimelineHTTPServer",
    "TimelineRequestHandler",
    "TimelineWebApp",
    "TimelineWebError",
    "create_server",
    "serve",
    "validate_report",
]
