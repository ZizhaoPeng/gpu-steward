"""Command-line integration for the GPU Steward Observe Plane."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .aggregate import build_day
from .codex import ingest_hook
from .collector import CollectorLoop
from .config import (
    DEFAULT_TIMELINE_CONFIG_PATH,
    GPUHostConfig,
    TimelineConfig,
    default_config_path,
)
from .gpu import GPUProbe
from .launchd import (
    DASHBOARD_PORT,
    DEFAULT_DASHBOARD_LABEL,
    DashboardLaunchAgentSpec,
    LaunchAgentSpec,
    LaunchdController,
    default_plist_path,
    default_dashboard_plist_path,
    install_dashboard_launch_agent,
    install_launch_agent,
    uninstall_dashboard_launch_agent,
    uninstall_launch_agent,
)
from .phases import CODEX_PHASES
from .store import TimelineStore
from .web import LOCALHOST, serve


class TimelineCLIError(ValueError):
    """Expected, user-facing timeline CLI failure."""


DASHBOARD_URL = "http://127.0.0.1:{}/".format(DASHBOARD_PORT)
DASHBOARD_HEALTH_URL = "http://127.0.0.1:{}/api/health".format(DASHBOARD_PORT)
DASHBOARD_START_TIMEOUT_SECONDS = 5.0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    timeline = subparsers.add_parser(
        "timeline", help="observe Codex phases and GPU activity without scheduling work"
    )
    commands = timeline.add_subparsers(dest="timeline_command", required=True)

    init = commands.add_parser("init", help="write a private local collector configuration")
    init.add_argument("--config", default=DEFAULT_TIMELINE_CONFIG_PATH)
    init.add_argument("--timeline-db", default=None)
    init.add_argument("--host", action="append", default=[])
    init.add_argument("--project", default=None)
    init.add_argument("--disabled-gpu", type=int, action="append", default=[])
    init.add_argument("--force", action="store_true")

    hook = commands.add_parser("hook", help=argparse.SUPPRESS)
    hook.add_argument("--timeline-db", default=None)

    phase = commands.add_parser("phase", help="declare one observable Codex work phase")
    phase.add_argument("phase", choices=CODEX_PHASES)
    phase.add_argument("--timeline-db", default=None)
    phase.add_argument("--project", default=None)

    sample = commands.add_parser("sample", help="perform one read-only GPU sample")
    sample.add_argument("--config", default=DEFAULT_TIMELINE_CONFIG_PATH)
    sample.add_argument("--json", action="store_true")

    collect = commands.add_parser("collect-loop", help="run token-free background sampling")
    collect.add_argument("--config", default=DEFAULT_TIMELINE_CONFIG_PATH)
    collect.add_argument("--max-iterations", type=int, default=None, help=argparse.SUPPRESS)

    report = commands.add_parser("report", help="export one local-day report")
    report.add_argument("--config", default=DEFAULT_TIMELINE_CONFIG_PATH)
    report.add_argument("--timeline-db", default=None)
    report.add_argument("--date", default=None)
    report.add_argument("--project", default=None)
    report.add_argument("--format", choices=("json", "csv"), default="json")
    report.add_argument("--output", default="-")

    dashboard = commands.add_parser("serve", help="serve the local read-only dashboard")
    dashboard.add_argument("--config", default=DEFAULT_TIMELINE_CONFIG_PATH)
    dashboard.add_argument("--timeline-db", default=None)
    dashboard.add_argument("--project", default=None)
    dashboard.add_argument("--port", type=int, default=8765)

    open_dashboard = commands.add_parser(
        "open", help="ensure the local dashboard is running and open it in the default browser"
    )
    open_dashboard.add_argument("--config", default=DEFAULT_TIMELINE_CONFIG_PATH)
    open_dashboard.add_argument("--plist", default=None)
    open_dashboard.add_argument("--executable", default=None)
    open_dashboard.add_argument("--project", default=None)

    dashboard_agent = commands.add_parser("dashboard", help="manage the macOS user dashboard")
    dashboard_agent.add_argument(
        "action", choices=("install", "start", "stop", "status", "uninstall")
    )
    dashboard_agent.add_argument("--config", default=DEFAULT_TIMELINE_CONFIG_PATH)
    dashboard_agent.add_argument("--plist", default=None)
    dashboard_agent.add_argument("--executable", default=None)
    dashboard_agent.add_argument("--project", default=None)

    collector = commands.add_parser("collector", help="manage the macOS user collector")
    collector.add_argument("action", choices=("install", "start", "stop", "status", "uninstall"))
    collector.add_argument("--config", default=DEFAULT_TIMELINE_CONFIG_PATH)
    collector.add_argument("--plist", default=None)
    collector.add_argument("--executable", default=None)
    return timeline


def _expanded(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _resolve_executable(executable: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve a fixed executable/module invocation for a LaunchAgent."""

    candidate = executable or shutil.which("gpu-steward")
    module_name = None
    pythonpath = None
    if candidate is None:
        candidate = os.path.abspath(sys.executable)
        module_name = "gpu_steward.cli"
        pythonpath = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    if not os.path.isabs(candidate):
        raise TimelineCLIError("cannot resolve the gpu-steward executable")
    return _expanded(candidate), module_name, pythonpath


def _dashboard_spec(
    *,
    config_path: str,
    plist_path: Optional[str] = None,
    executable: Optional[str] = None,
    project: Optional[str] = None,
) -> DashboardLaunchAgentSpec:
    config_path = _expanded(config_path)
    # Launchd opens these paths itself; make the private directory now so a
    # first-run dashboard does not fail merely because the config is absent.
    config_parent = os.path.dirname(config_path)
    os.makedirs(config_parent, mode=0o700, exist_ok=True)
    resolved_executable, module_name, pythonpath = _resolve_executable(executable)
    return DashboardLaunchAgentSpec(
        executable=resolved_executable,
        config_path=config_path,
        log_path=os.path.join(config_parent, "timeline-dashboard.log"),
        plist_path=_expanded(plist_path) if plist_path else None,
        project=project,
        module_name=module_name,
        pythonpath=pythonpath,
    )


def _install_dashboard_agent(
    *,
    config_path: str,
    plist_path: Optional[str] = None,
    executable: Optional[str] = None,
    project: Optional[str] = None,
) -> Dict[str, Any]:
    spec = _dashboard_spec(
        config_path=config_path,
        plist_path=plist_path,
        executable=executable,
        project=project,
    )
    installed = install_dashboard_launch_agent(spec, spec.resolved_plist_path)
    controller = LaunchdController(label=DEFAULT_DASHBOARD_LABEL)
    # Installation is also the upgrade path. Reload the service so launchd
    # uses the new executable/arguments instead of keeping an older process.
    started = controller.replace_started(installed)
    return {
        "schema_version": 1,
        "ok": True,
        "action": "installed",
        "label": DEFAULT_DASHBOARD_LABEL,
        "plist": installed,
        "started": bool(started),
        "url": DASHBOARD_URL,
    }


def _dashboard_available(timeout: float = 0.25) -> bool:
    """Return true only for a healthy GPU Steward dashboard on localhost."""

    try:
        response = urllib.request.urlopen(DASHBOARD_HEALTH_URL, timeout=timeout)
        try:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if int(status) != 200:
                return False
            raw = response.read(4097)
            if len(raw) > 4096:
                return False
            payload = json.loads(raw.decode("utf-8"))
            return bool(
                isinstance(payload, Mapping)
                and payload.get("ok") is True
                and payload.get("service") == "gpu-steward-timeline"
                and payload.get("plane") == "observe"
            )
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return False


def _wait_for_dashboard(timeout: float = DASHBOARD_START_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        if _dashboard_available():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _private_write(path: str, text: str, force: bool = False) -> None:
    target = _expanded(path)
    if os.path.exists(target) and not force:
        raise TimelineCLIError("refusing to overwrite existing file; pass --force")
    parent = os.path.dirname(target)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".gpu-steward.", dir=parent, text=True)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_config(path: str, *, allow_missing: bool = False) -> TimelineConfig:
    target = _expanded(path)
    if allow_missing and not os.path.exists(target):
        return TimelineConfig()
    return TimelineConfig.load(target)


def _today(timezone: str) -> str:
    if timezone == "Asia/Singapore":
        return (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).date().isoformat()
    return datetime.datetime.utcnow().date().isoformat()


def _store_path(args: argparse.Namespace, config: Optional[TimelineConfig] = None) -> Optional[str]:
    return getattr(args, "timeline_db", None) or (config.database_path if config else None)


def _emit(payload: Mapping[str, Any], stream: Any = None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def _collector(config: TimelineConfig, store: TimelineStore) -> CollectorLoop:
    return CollectorLoop(
        [GPUProbe(host) for host in config.hosts if host.enabled],
        store,
        sample_interval_seconds=config.sample_interval_seconds,
        idle_sample_interval_seconds=config.idle_sample_interval_seconds,
        backoff_seconds=config.backoff_seconds,
    )


def _report_builder(store: TimelineStore):
    def builder(*, date: str, timezone: str, project: Optional[str]):
        return build_day(store, date, timezone=timezone, project=project)

    return builder


def _write_report(report: Mapping[str, Any], output: str, format_name: str) -> None:
    if output == "-":
        handle = sys.stdout
        close = False
    else:
        target = _expanded(output)
        os.makedirs(os.path.dirname(target), mode=0o700, exist_ok=True)
        handle = open(target, "w", encoding="utf-8", newline="")
        close = True
    try:
        if format_name == "json":
            json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            return
        writer = csv.writer(handle)
        writer.writerow(
            ["date", "timezone", "lane_id", "lane_kind", "lane_label", "start", "end", "state", "task_name", "source", "confidence"]
        )
        for lane in report.get("lanes", []):
            for segment in lane.get("segments", []):
                writer.writerow(
                    [
                        report.get("date", ""), report.get("timezone", ""), lane.get("id", ""),
                        lane.get("kind", ""), lane.get("label", ""), segment.get("start", ""),
                        segment.get("end", ""), segment.get("phase", segment.get("state", "")),
                        segment.get("task_name", ""), segment.get("source", ""), segment.get("confidence", ""),
                    ]
                )
    finally:
        if close:
            handle.close()


def dispatch(args: argparse.Namespace) -> int:
    command = args.timeline_command
    if command == "init":
        if any(index < 0 for index in args.disabled_gpu):
            raise TimelineCLIError("--disabled-gpu must be non-negative")
        hosts = tuple(
            GPUHostConfig(
                name=host,
                host=host,
                disabled_gpu_indices=tuple(args.disabled_gpu),
                project=args.project,
            )
            for host in args.host
        )
        config = TimelineConfig(
            hosts=hosts,
            database_path=args.timeline_db or "~/.gpu-steward/timeline.sqlite3",
        )
        target = _expanded(args.config)
        _private_write(target, config.to_json(), force=args.force)
        _emit({"schema_version": 1, "ok": True, "config": target, "hosts": len(hosts)})
        return 0

    if command == "hook":
        try:
            payload = json.load(sys.stdin)
        except (TypeError, ValueError) as exc:
            raise TimelineCLIError("hook input must be one JSON object") from exc
        store = TimelineStore(_store_path(args))
        try:
            ingest_hook(payload, store)
        finally:
            store.close()
        # Hook success is deliberately silent: it must not consume model context.
        return 0

    if command == "phase":
        store = TimelineStore(_store_path(args))
        try:
            now = time.time()
            project = args.project or os.getcwd()
            store.record_codex_event(
                event_id=None,
                occurred_at=now,
                session_id=os.environ.get("CODEX_SESSION_ID", "manual-session"),
                turn_id=os.environ.get("CODEX_TURN_ID", "manual-turn"),
                project=project,
                kind="phase",
                phase=args.phase,
                source="declared",
                confidence=1.0,
            )
        finally:
            store.close()
        _emit({"schema_version": 1, "ok": True, "phase": args.phase})
        return 0

    if command in ("sample", "collect-loop"):
        config = _load_config(args.config)
        store = TimelineStore(config.database_path)
        try:
            collector = _collector(config, store)
            if command == "sample":
                result = collector.collect_once()
                if args.json:
                    _emit(result.as_dict())
                return 0 if result.ok else 1
            collector.run_forever(max_iterations=args.max_iterations)
            return 0
        finally:
            store.close()

    if command == "report":
        config = _load_config(args.config, allow_missing=True)
        store = TimelineStore(_store_path(args, config))
        try:
            report = build_day(
                store,
                args.date or _today(config.timezone),
                timezone=config.timezone,
                project=args.project,
            )
        finally:
            store.close()
        _write_report(report, args.output, args.format)
        return 0

    if command == "serve":
        config = _load_config(args.config, allow_missing=True)
        store = TimelineStore(_store_path(args, config))
        try:
            serve(
                _report_builder(store),
                host=LOCALHOST,
                port=args.port,
                timezone=config.timezone,
                project=args.project,
            )
        finally:
            store.close()
        return 0

    if command == "open":
        # A manually started or already-installed dashboard is reused when its
        # health endpoint answers.  This is the fast path and guarantees that
        # opening the UI does not create another process on port 8765.
        started = False
        if not _dashboard_available():
            payload = _install_dashboard_agent(
                config_path=args.config,
                plist_path=args.plist,
                executable=args.executable,
                project=args.project,
            )
            started = bool(payload.get("started"))
            if not _wait_for_dashboard():
                raise TimelineCLIError("dashboard did not become available on localhost:8765")
        try:
            browser_opened = bool(webbrowser.open(DASHBOARD_URL, new=0, autoraise=True))
        except Exception as exc:
            raise TimelineCLIError("cannot open the default browser") from exc
        if not browser_opened:
            raise TimelineCLIError("default browser did not accept the dashboard URL")
        _emit(
            {
                "schema_version": 1,
                "ok": True,
                "action": "opened",
                "url": DASHBOARD_URL,
                "started": started,
                "browser_opened": browser_opened,
            }
        )
        return 0

    if command == "dashboard":
        plist_path = _expanded(args.plist or default_dashboard_plist_path())
        controller = LaunchdController(label=DEFAULT_DASHBOARD_LABEL)
        if args.action == "install":
            payload = _install_dashboard_agent(
                config_path=args.config,
                plist_path=plist_path,
                executable=args.executable,
                project=args.project,
            )
            _emit(payload)
            return 0
        if args.action == "start":
            started = controller.ensure_started(plist_path)
            _emit(
                {
                    "schema_version": 1,
                    "ok": True,
                    "action": "started",
                    "plist": plist_path,
                    "started": bool(started),
                    "url": DASHBOARD_URL,
                }
            )
            return 0
        if args.action == "stop":
            controller.stop()
            _emit({"schema_version": 1, "ok": True, "action": "stopped"})
            return 0
        if args.action == "status":
            controller.status()
            _emit({"schema_version": 1, "ok": True, "action": "status", "url": DASHBOARD_URL})
            return 0
        if args.action == "uninstall":
            try:
                controller.stop()
            except Exception:
                # The service may already be unloaded; deleting only our
                # labelled plist remains safe and reversible.
                pass
            removed = uninstall_dashboard_launch_agent(plist_path)
            _emit(
                {
                    "schema_version": 1,
                    "ok": True,
                    "action": "uninstalled",
                    "removed": removed,
                }
            )
            return 0
        raise TimelineCLIError("unknown dashboard action")

    if command == "collector":
        config_path = _expanded(args.config)
        plist_path = _expanded(args.plist or default_plist_path())
        controller = LaunchdController()
        if args.action == "install":
            _load_config(config_path)
            executable, module_name, pythonpath = _resolve_executable(args.executable)
            log_path = os.path.join(os.path.dirname(config_path), "timeline-collector.log")
            spec = LaunchAgentSpec(
                executable,
                config_path,
                log_path,
                plist_path=plist_path,
                module_name=module_name,
                pythonpath=pythonpath,
            )
            installed = install_launch_agent(spec, plist_path)
            # Collector installation also upgrades its runtime. Reloading is
            # intentional and does not touch any scheduled/training process.
            started = controller.replace_started(installed)
            _emit(
                {
                    "schema_version": 1,
                    "ok": True,
                    "action": "installed",
                    "plist": installed,
                    "started": bool(started),
                }
            )
            return 0
        if args.action == "start":
            controller.start(plist_path)
        elif args.action == "stop":
            controller.stop()
        elif args.action == "status":
            controller.status()
        elif args.action == "uninstall":
            try:
                controller.stop()
            except Exception:
                pass
            removed = uninstall_launch_agent(plist_path)
            _emit({"schema_version": 1, "ok": True, "action": "uninstalled", "removed": removed})
            return 0
        _emit({"schema_version": 1, "ok": True, "action": args.action})
        return 0

    raise TimelineCLIError("unknown timeline command")


__all__ = ["TimelineCLIError", "add_parser", "dispatch"]
