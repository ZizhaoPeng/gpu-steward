#!/usr/bin/env python3
"""Local bootstrap for the GPU Steward Observe Plane.

This helper keeps the timeline setup token-free:

- writes a private local timeline config;
- links the repo plugin into the default personal marketplace root;
- installs/reinstalls the plugin through the local Codex CLI;
- installs the macOS user LaunchAgent collector and dashboard.

It is intentionally local-only and never touches remote training state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


PLUGIN_NAME = "gpu-steward-timeline"
DEFAULT_MARKETPLACE_NAME = "personal"
DEFAULT_CATEGORY = "Productivity"
DEFAULT_CONFIG_PATH = Path("~/.gpu-steward/timeline.json").expanduser()
DEFAULT_MARKETPLACE_PATH = Path("~/.agents/plugins/marketplace.json").expanduser()
PLUGIN_CREATOR = Path("~/.codex/skills/.system/plugin-creator/scripts/create_basic_plugin.py").expanduser()
DEFAULT_RUNTIME_VENV = Path("~/.gpu-steward/venv").expanduser()


class BootstrapError(RuntimeError):
    """Expected local bootstrap failure."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_plugin_source() -> Path:
    return repo_root() / "integrations" / "codex" / PLUGIN_NAME


def default_local_plugin_path() -> Path:
    return Path.home() / "plugins" / PLUGIN_NAME


def _env_with_pythonpath(root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    src_path = str(root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return env


def _run(argv: List[str], *, cwd: Path, env: Optional[Mapping[str, str]] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd),
            env=None if env is None else dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise BootstrapError("cannot execute local command: {}".format(argv[0])) from exc


def _run_checked(
    argv: List[str],
    *,
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
    generic_error: str,
) -> subprocess.CompletedProcess:
    result = _run(argv, cwd=cwd, env=env)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or generic_error
        raise BootstrapError(message)
    return result


def ensure_symlink(target: Path, source: Path, *, force: bool = False) -> Path:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source:
            return target
        if not force:
            raise BootstrapError("plugin target already exists: {}".format(target))
        if target.is_dir() and not target.is_symlink():
            raise BootstrapError("refusing to replace a real directory: {}".format(target))
        target.unlink()
    target.symlink_to(source, target_is_directory=True)
    return target


def scaffold_marketplace(root: Path, path: Path, *, force: bool) -> str:
    """Use the official plugin scaffold to create/update marketplace metadata."""

    if not PLUGIN_CREATOR.exists():
        raise BootstrapError("Codex plugin creator is unavailable")
    argv = [
        sys.executable,
        str(PLUGIN_CREATOR),
        PLUGIN_NAME,
        "--path",
        str(DEFAULT_CONFIG_PATH.parent / "plugin-scaffold"),
        "--marketplace-path",
        str(path),
        "--with-marketplace",
        "--with-skills",
        "--with-hooks",
        "--with-scripts",
        "--category",
        DEFAULT_CATEGORY,
    ]
    if path.exists() or force:
        argv.append("--force")
    _run_checked(argv, cwd=root, generic_error="plugin marketplace scaffold failed")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        name = payload["name"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BootstrapError("cannot read scaffolded personal marketplace") from exc
    if not isinstance(name, str) or not name.strip():
        raise BootstrapError("personal marketplace needs a non-empty name")
    return name


def timeline_command(root: Path, *args: str) -> List[str]:
    return [sys.executable, "-m", "gpu_steward.cli"] + list(args)


def write_timeline_config(
    root: Path,
    *,
    config_path: Path,
    timeline_db: Optional[Path],
    hosts: Iterable[str],
    project: Optional[str],
    disabled_gpu: Iterable[int],
    force: bool,
) -> Dict[str, Any]:
    argv = timeline_command(root, "timeline", "init", "--config", str(config_path))
    if timeline_db is not None:
        argv.extend(["--timeline-db", str(timeline_db)])
    for host in hosts:
        argv.extend(["--host", host])
    if project:
        argv.extend(["--project", project])
    for index in disabled_gpu:
        argv.extend(["--disabled-gpu", str(index)])
    if force:
        argv.append("--force")
    result = _run_checked(
        argv,
        cwd=root,
        env=_env_with_pythonpath(root),
        generic_error="timeline init failed",
    )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise BootstrapError("timeline init returned invalid JSON") from exc


def install_plugin(root: Path, *, marketplace_path: Path, plugin_source: Path, force: bool) -> Dict[str, str]:
    if not plugin_source.exists():
        raise BootstrapError("plugin source does not exist: {}".format(plugin_source))
    linked = ensure_symlink(
        default_local_plugin_path(),
        plugin_source,
        force=force,
    )
    marketplace_name = scaffold_marketplace(root, marketplace_path, force=force)
    _run_checked(
        ["codex", "plugin", "add", "{}@{}".format(PLUGIN_NAME, marketplace_name)],
        cwd=root,
        generic_error="codex plugin add failed",
    )
    return {
        "marketplace_name": marketplace_name,
        "marketplace_path": str(marketplace_path),
        "plugin_link": str(linked),
    }


def install_runtime(root: Path, venv_path: Path) -> Path:
    python = venv_path / "bin" / "python"
    executable = venv_path / "bin" / "gpu-steward"
    if not python.exists():
        _run_checked(
            [sys.executable, "-m", "venv", str(venv_path)],
            cwd=root,
            generic_error="cannot create private GPU Steward runtime",
        )
    _run_checked(
        [str(python), "-m", "pip", "install", "--no-deps", str(root)],
        cwd=root,
        generic_error="cannot install private GPU Steward runtime",
    )
    if not executable.exists():
        raise BootstrapError("private GPU Steward runtime has no executable")
    return executable.resolve()


def install_collector(root: Path, *, config_path: Path, executable: Path) -> Dict[str, Any]:
    result = _run_checked(
        [str(executable), "timeline", "collector", "install", "--config", str(config_path), "--executable", str(executable)],
        cwd=root,
        generic_error="timeline collector install failed",
    )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise BootstrapError("collector install returned invalid JSON") from exc


def install_dashboard(
    root: Path,
    *,
    config_path: Path,
    executable: Path,
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """Install the persistent localhost dashboard LaunchAgent."""

    argv = [
        str(executable),
        "timeline",
        "dashboard",
        "install",
        "--config",
        str(config_path),
        "--executable",
        str(executable),
    ]
    if project:
        argv.extend(["--project", project])
    result = _run_checked(
        argv,
        cwd=root,
        generic_error="timeline dashboard install failed",
    )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise BootstrapError("dashboard install returned invalid JSON") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the local GPU Steward timeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--timeline-db", default=None)
    parser.add_argument("--host", action="append", required=True)
    parser.add_argument("--project", default=None)
    parser.add_argument("--disabled-gpu", type=int, action="append", default=[])
    parser.add_argument("--marketplace-path", default=str(DEFAULT_MARKETPLACE_PATH))
    parser.add_argument("--plugin-source", default=str(default_plugin_source()))
    parser.add_argument("--runtime-venv", default=str(DEFAULT_RUNTIME_VENV))
    parser.add_argument("--skip-plugin", action="store_true")
    parser.add_argument("--skip-launchd", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    config_path = Path(args.config).expanduser().resolve()
    timeline_db = None if args.timeline_db is None else Path(args.timeline_db).expanduser().resolve()
    plugin_source = Path(args.plugin_source).expanduser().resolve()
    marketplace_path = Path(args.marketplace_path).expanduser().resolve()
    runtime_venv = Path(args.runtime_venv).expanduser().resolve()

    try:
        config_payload = write_timeline_config(
            root,
            config_path=config_path,
            timeline_db=timeline_db,
            hosts=args.host,
            project=args.project,
            disabled_gpu=args.disabled_gpu,
            force=args.force,
        )
        plugin_payload: Optional[Dict[str, str]] = None
        if not args.skip_plugin:
            plugin_payload = install_plugin(
                root,
                marketplace_path=marketplace_path,
                plugin_source=plugin_source,
                force=args.force,
            )
        collector_payload: Optional[Dict[str, Any]] = None
        dashboard_payload: Optional[Dict[str, Any]] = None
        if not args.skip_launchd:
            executable = install_runtime(root, runtime_venv)
            collector_payload = install_collector(
                root, config_path=config_path, executable=executable
            )
            # Keep the dashboard unfiltered by default so the daily view still
            # shows every GPU occupant; callers can use ``timeline open
            # --project`` when a focused Codex lane is desired.
            dashboard_payload = install_dashboard(
                root, config_path=config_path, executable=executable
            )
    except BootstrapError as exc:
        sys.stderr.write(str(exc).strip() + "\n")
        return 2

    payload: Dict[str, Any] = {
        "ok": True,
        "config": str(config_path),
        "hosts": list(args.host),
        "dashboard_url": "http://127.0.0.1:8765/",
        "timeline_db": None if timeline_db is None else str(timeline_db),
        "timeline_init": config_payload,
    }
    if plugin_payload is not None:
        payload["plugin"] = plugin_payload
    if collector_payload is not None:
        payload["collector"] = collector_payload
    if dashboard_payload is not None:
        payload["dashboard"] = dashboard_payload
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
