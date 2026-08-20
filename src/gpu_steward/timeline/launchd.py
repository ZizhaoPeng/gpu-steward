"""Reversible macOS user LaunchAgent helpers for ``collect-loop``.

The helpers render and optionally manage one user-level plist.  They never
write a LaunchAgent during import and never invoke a shell.  Tests and
callers can provide an explicit temporary plist path and a command runner;
the default path is only calculated, not created.
"""

import os
import plistlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


DEFAULT_LABEL = "com.gpu-steward.timeline.collector"
DEFAULT_PLIST_NAME = DEFAULT_LABEL + ".plist"
DEFAULT_DASHBOARD_LABEL = "com.gpu-steward.timeline.dashboard"
DEFAULT_DASHBOARD_PLIST_NAME = DEFAULT_DASHBOARD_LABEL + ".plist"
DASHBOARD_PORT = 8765
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


class LaunchAgentError(RuntimeError):
    """Raised when rendering or controlling a user LaunchAgent fails."""


def default_launch_agents_dir(home: Optional[str] = None) -> str:
    base = os.path.expanduser(home or "~")
    return os.path.join(base, "Library", "LaunchAgents")


def default_plist_path(home: Optional[str] = None) -> str:
    return os.path.join(default_launch_agents_dir(home), DEFAULT_PLIST_NAME)


def default_dashboard_plist_path(home: Optional[str] = None) -> str:
    return os.path.join(default_launch_agents_dir(home), DEFAULT_DASHBOARD_PLIST_NAME)


def _absolute_path(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LaunchAgentError("{} must be a non-empty path".format(field_name))
    path = os.path.abspath(os.path.expanduser(value))
    if any(ord(char) < 32 for char in path):
        raise LaunchAgentError("{} contains a control character".format(field_name))
    return path


def _safe_executable(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LaunchAgentError("executable must be an absolute path")
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        raise LaunchAgentError("executable must be absolute")
    path = _absolute_path(value, "executable")
    if not os.path.isabs(path):
        raise LaunchAgentError("executable must be absolute")
    return path


@dataclass(frozen=True)
class LaunchAgentSpec:
    """Canonical user LaunchAgent configuration."""

    executable: str
    config_path: str
    log_path: str
    plist_path: Optional[str] = None
    label: str = DEFAULT_LABEL
    interval_seconds: int = 30
    module_name: Optional[str] = None
    pythonpath: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "executable", _safe_executable(self.executable))
        object.__setattr__(self, "config_path", _absolute_path(self.config_path, "config_path"))
        object.__setattr__(self, "log_path", _absolute_path(self.log_path, "log_path"))
        if self.plist_path is not None:
            object.__setattr__(self, "plist_path", _absolute_path(self.plist_path, "plist_path"))
        if not isinstance(self.label, str) or not self.label or any(
            char.isspace() or char in "\x00/" for char in self.label
        ):
            raise LaunchAgentError("invalid LaunchAgent label")
        if self.module_name is not None:
            if not isinstance(self.module_name, str) or not _MODULE_RE.match(self.module_name):
                raise LaunchAgentError("invalid Python module name")
        if self.pythonpath is not None:
            object.__setattr__(self, "pythonpath", _absolute_path(self.pythonpath, "pythonpath"))
        try:
            interval = int(self.interval_seconds)
        except (TypeError, ValueError) as exc:
            raise LaunchAgentError("interval_seconds must be an integer") from exc
        if interval < 1:
            raise LaunchAgentError("interval_seconds must be positive")
        object.__setattr__(self, "interval_seconds", interval)

    @property
    def resolved_plist_path(self) -> str:
        return self.plist_path or default_plist_path()

    @property
    def program_arguments(self) -> List[str]:
        # Keep this fixed.  The plist must not become an arbitrary command
        # runner or carry command arguments from a training job.
        if self.module_name is None:
            return [
                self.executable,
                "timeline",
                "collect-loop",
                "--config",
                self.config_path,
            ]
        return [
            self.executable,
            "-m",
            self.module_name,
            "timeline",
            "collect-loop",
            "--config",
            self.config_path,
        ]

    def mapping(self) -> Dict[str, Any]:
        result = {
            "Label": self.label,
            "ProgramArguments": self.program_arguments,
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Background",
            "LowPriorityIO": True,
            "StandardOutPath": self.log_path,
            "StandardErrorPath": self.log_path,
        }
        if self.pythonpath is not None:
            result["EnvironmentVariables"] = {"PYTHONPATH": self.pythonpath}
        return result


@dataclass(frozen=True)
class DashboardLaunchAgentSpec:
    """Canonical user LaunchAgent configuration for the local dashboard.

    The dashboard is a separate service from the collector so opening the UI
    never creates a second collector process.  Its port and command shape are
    fixed by the Observe Plane contract; only the executable and private
    configuration paths are supplied by the caller.
    """

    executable: str
    config_path: str
    log_path: str
    plist_path: Optional[str] = None
    label: str = DEFAULT_DASHBOARD_LABEL
    project: Optional[str] = None
    module_name: Optional[str] = None
    pythonpath: Optional[str] = None
    port: int = DASHBOARD_PORT

    def __post_init__(self):
        object.__setattr__(self, "executable", _safe_executable(self.executable))
        object.__setattr__(self, "config_path", _absolute_path(self.config_path, "config_path"))
        object.__setattr__(self, "log_path", _absolute_path(self.log_path, "log_path"))
        if self.plist_path is not None:
            object.__setattr__(self, "plist_path", _absolute_path(self.plist_path, "plist_path"))
        if not isinstance(self.label, str) or not self.label or any(
            char.isspace() or char in "\x00/" for char in self.label
        ):
            raise LaunchAgentError("invalid LaunchAgent label")
        if self.project is not None:
            if not isinstance(self.project, str) or not self.project.strip():
                raise LaunchAgentError("project must be a non-empty string")
            if "\x00" in self.project or any(ord(char) < 32 for char in self.project):
                raise LaunchAgentError("project contains a control character")
            if len(self.project) > 256:
                raise LaunchAgentError("project is too long")
            object.__setattr__(self, "project", self.project.strip())
        if self.module_name is not None:
            if not isinstance(self.module_name, str) or not _MODULE_RE.match(self.module_name):
                raise LaunchAgentError("invalid Python module name")
        if self.pythonpath is not None:
            object.__setattr__(self, "pythonpath", _absolute_path(self.pythonpath, "pythonpath"))
        try:
            port = int(self.port)
        except (TypeError, ValueError) as exc:
            raise LaunchAgentError("dashboard port must be an integer") from exc
        if port != DASHBOARD_PORT:
            raise LaunchAgentError("dashboard port is fixed at 8765")
        object.__setattr__(self, "port", port)

    @property
    def resolved_plist_path(self) -> str:
        return self.plist_path or default_dashboard_plist_path()

    @property
    def program_arguments(self) -> List[str]:
        command = [
            "timeline",
            "serve",
            "--config",
            self.config_path,
            "--port",
            str(self.port),
        ]
        if self.project is not None:
            command.extend(["--project", self.project])
        if self.module_name is None:
            return [self.executable] + command
        return [self.executable, "-m", self.module_name] + command

    def mapping(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "Label": self.label,
            "ProgramArguments": self.program_arguments,
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Background",
            "LowPriorityIO": True,
            "StandardOutPath": self.log_path,
            "StandardErrorPath": self.log_path,
        }
        if self.pythonpath is not None:
            result["EnvironmentVariables"] = {"PYTHONPATH": self.pythonpath}
        return result


# Short spelling retained for callers that refer to the collector's
# ``LaunchAgentSpec`` as an ``AgentSpec``.
DashboardAgentSpec = DashboardLaunchAgentSpec


def render_plist(
    executable: str,
    config_path: str,
    log_path: str,
    *,
    label: str = DEFAULT_LABEL,
    interval_seconds: int = 30,
    plist_path: Optional[str] = None,
    module_name: Optional[str] = None,
    pythonpath: Optional[str] = None,
) -> bytes:
    """Render deterministic XML bytes without writing a file."""

    spec = LaunchAgentSpec(
        executable=executable,
        config_path=config_path,
        log_path=log_path,
        label=label,
        interval_seconds=interval_seconds,
        plist_path=plist_path,
        module_name=module_name,
        pythonpath=pythonpath,
    )
    return plistlib.dumps(spec.mapping(), fmt=plistlib.FMT_XML, sort_keys=False)


def render_launch_agent(spec: LaunchAgentSpec) -> bytes:
    if not isinstance(spec, LaunchAgentSpec):
        raise LaunchAgentError("spec must be LaunchAgentSpec")
    return plistlib.dumps(spec.mapping(), fmt=plistlib.FMT_XML, sort_keys=False)


def render_dashboard_launch_agent(spec: DashboardLaunchAgentSpec) -> bytes:
    if not isinstance(spec, DashboardLaunchAgentSpec):
        raise LaunchAgentError("spec must be DashboardLaunchAgentSpec")
    return plistlib.dumps(spec.mapping(), fmt=plistlib.FMT_XML, sort_keys=False)


def render_dashboard_plist(
    executable: str,
    config_path: str,
    log_path: str,
    *,
    label: str = DEFAULT_DASHBOARD_LABEL,
    project: Optional[str] = None,
    plist_path: Optional[str] = None,
    module_name: Optional[str] = None,
    pythonpath: Optional[str] = None,
    port: int = DASHBOARD_PORT,
) -> bytes:
    """Render deterministic XML for the fixed localhost dashboard service."""

    spec = DashboardLaunchAgentSpec(
        executable=executable,
        config_path=config_path,
        log_path=log_path,
        label=label,
        project=project,
        plist_path=plist_path,
        module_name=module_name,
        pythonpath=pythonpath,
        port=port,
    )
    return render_dashboard_launch_agent(spec)


def _write_private_atomic(path: str, payload: bytes):
    parent = os.path.dirname(path)
    if not parent:
        raise LaunchAgentError("plist path has no parent directory")
    try:
        os.makedirs(parent, mode=0o700, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".gpu-steward.", suffix=".plist", dir=parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise LaunchAgentError("cannot write LaunchAgent plist") from exc


def install_launch_agent(spec: LaunchAgentSpec, path: Optional[str] = None) -> str:
    """Write one user plist and return its path.

    The caller must pass an explicit ``path`` for tests or non-default
    locations.  No ``launchctl`` command is run; activation remains a separate
    explicit operation and can be reversed with :func:`uninstall_launch_agent`.
    """

    if not isinstance(spec, LaunchAgentSpec):
        raise LaunchAgentError("spec must be LaunchAgentSpec")
    target = _absolute_path(path or spec.resolved_plist_path, "plist_path")
    if os.path.exists(target):
        # Updating our own plist is allowed; clobbering another user's
        # LaunchAgent would make the helper non-reversible.
        if _read_spec_label(target) != spec.label:
            raise LaunchAgentError("refusing to overwrite an unrelated LaunchAgent")
    payload = render_launch_agent(spec)
    _write_private_atomic(target, payload)
    return target


def install_dashboard_launch_agent(
    spec: DashboardLaunchAgentSpec, path: Optional[str] = None
) -> str:
    """Write one private dashboard plist and return its path."""

    if not isinstance(spec, DashboardLaunchAgentSpec):
        raise LaunchAgentError("spec must be DashboardLaunchAgentSpec")
    target = _absolute_path(path or spec.resolved_plist_path, "plist_path")
    if os.path.exists(target):
        if _read_spec_label(target) != spec.label:
            raise LaunchAgentError("refusing to overwrite an unrelated LaunchAgent")
    payload = render_dashboard_launch_agent(spec)
    _write_private_atomic(target, payload)
    return target


def _read_spec_label(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            value = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        raise LaunchAgentError("cannot read LaunchAgent plist") from exc
    return value.get("Label") if isinstance(value, Mapping) else None


def uninstall_launch_agent(path: Optional[str] = None, expected_label: str = DEFAULT_LABEL) -> bool:
    """Remove only the expected timeline plist; return whether it existed."""

    target = _absolute_path(path or default_plist_path(), "plist_path")
    if not os.path.exists(target):
        return False
    label = _read_spec_label(target)
    if label != expected_label:
        raise LaunchAgentError("refusing to remove an unrelated LaunchAgent")
    try:
        os.unlink(target)
    except OSError as exc:
        raise LaunchAgentError("cannot remove LaunchAgent plist") from exc
    return True


def uninstall_dashboard_launch_agent(
    path: Optional[str] = None, expected_label: str = DEFAULT_DASHBOARD_LABEL
) -> bool:
    """Remove only the expected dashboard plist; return whether it existed."""

    target = _absolute_path(path or default_dashboard_plist_path(), "plist_path")
    if not os.path.exists(target):
        return False
    label = _read_spec_label(target)
    if label != expected_label:
        raise LaunchAgentError("refusing to remove an unrelated LaunchAgent")
    try:
        os.unlink(target)
    except OSError as exc:
        raise LaunchAgentError("cannot remove LaunchAgent plist") from exc
    return True


Runner = Callable[[Sequence[str]], Any]


def _default_runner(argv: Sequence[str]) -> Any:
    try:
        return subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchAgentError("launchctl command failed") from exc


class LaunchdController:
    """Build and optionally execute user-scoped ``launchctl`` argv calls."""

    def __init__(
        self,
        label: str = DEFAULT_LABEL,
        uid: Optional[int] = None,
        runner: Optional[Runner] = None,
    ):
        if not isinstance(label, str) or not label or any(char.isspace() for char in label):
            raise LaunchAgentError("invalid LaunchAgent label")
        self.label = label
        self.uid = os.getuid() if uid is None else int(uid)
        if self.uid < 1:
            raise LaunchAgentError("LaunchAgent control requires a non-root user")
        self.runner = runner or _default_runner

    @property
    def domain(self) -> str:
        return "gui/{}".format(self.uid)

    @property
    def service_target(self) -> str:
        return "{}/{}".format(self.domain, self.label)

    def command(self, action: str, plist_path: Optional[str] = None) -> List[str]:
        target = _absolute_path(plist_path, "plist_path") if plist_path else None
        if action == "start":
            if target is None:
                raise LaunchAgentError("start requires a plist path")
            return ["launchctl", "bootstrap", self.domain, target]
        if action == "stop":
            return ["launchctl", "bootout", self.service_target]
        if action == "restart":
            if target is None:
                raise LaunchAgentError("restart requires a plist path")
            return ["launchctl", "kickstart", "-k", self.service_target]
        if action == "status":
            return ["launchctl", "print", self.service_target]
        raise LaunchAgentError("unknown LaunchAgent action")

    def execute(self, action: str, plist_path: Optional[str] = None) -> Any:
        argv = self.command(action, plist_path)
        try:
            result = self.runner(list(argv))
        except LaunchAgentError:
            raise
        except Exception as exc:
            raise LaunchAgentError("launchctl command failed") from exc
        if getattr(result, "returncode", None) != 0:
            raise LaunchAgentError("launchctl command returned a non-zero status")
        return result

    def start(self, plist_path: str) -> Any:
        return self.execute("start", plist_path)

    def ensure_started(self, plist_path: str) -> bool:
        """Ensure this service is loaded once; return whether it was started.

        ``launchctl bootstrap`` is intentionally not called when the service
        is already loaded.  This makes repeated ``timeline open`` and
        bootstrap invocations idempotent and avoids duplicate dashboard
        processes.  A small status/start/status retry also handles the race
        where another invocation loads the same user service concurrently.
        """

        try:
            self.status()
            return False
        except LaunchAgentError:
            pass
        try:
            self.start(plist_path)
            return True
        except LaunchAgentError:
            try:
                self.status()
                return False
            except LaunchAgentError:
                raise

    def replace_started(self, plist_path: str) -> bool:
        """Reload a service from its updated plist and leave it running."""

        try:
            self.stop()
        except LaunchAgentError:
            pass
        self.start(plist_path)
        return True

    def stop(self) -> Any:
        return self.execute("stop")

    def restart(self) -> Any:
        return self.execute("restart")

    def status(self) -> Any:
        return self.execute("status")


Launchd = LaunchdController


def launchctl_command(action: str, label: str = DEFAULT_LABEL, uid: Optional[int] = None, plist_path: Optional[str] = None) -> List[str]:
    return LaunchdController(label=label, uid=uid).command(action, plist_path)


# Short procedural spellings are useful to a tiny CLI wrapper while keeping
# all filesystem and launchctl behavior in the explicit helpers above.
def install(spec: LaunchAgentSpec, path: Optional[str] = None) -> str:
    return install_launch_agent(spec, path)


def install_dashboard(spec: DashboardLaunchAgentSpec, path: Optional[str] = None) -> str:
    return install_dashboard_launch_agent(spec, path)


def uninstall(path: Optional[str] = None, expected_label: str = DEFAULT_LABEL) -> bool:
    return uninstall_launch_agent(path, expected_label)


def uninstall_dashboard(
    path: Optional[str] = None, expected_label: str = DEFAULT_DASHBOARD_LABEL
) -> bool:
    return uninstall_dashboard_launch_agent(path, expected_label)


def start(path: str, label: str = DEFAULT_LABEL, uid: Optional[int] = None) -> Any:
    return LaunchdController(label=label, uid=uid).start(path)


def stop(label: str = DEFAULT_LABEL, uid: Optional[int] = None) -> Any:
    return LaunchdController(label=label, uid=uid).stop()


def status(label: str = DEFAULT_LABEL, uid: Optional[int] = None) -> Any:
    return LaunchdController(label=label, uid=uid).status()


__all__ = [
    "DASHBOARD_PORT",
    "DEFAULT_LABEL",
    "DEFAULT_PLIST_NAME",
    "DEFAULT_DASHBOARD_LABEL",
    "DEFAULT_DASHBOARD_PLIST_NAME",
    "LaunchAgentError",
    "LaunchAgentSpec",
    "DashboardAgentSpec",
    "DashboardLaunchAgentSpec",
    "Launchd",
    "LaunchdController",
    "default_launch_agents_dir",
    "default_dashboard_plist_path",
    "default_plist_path",
    "install",
    "install_dashboard",
    "install_dashboard_launch_agent",
    "install_launch_agent",
    "launchctl_command",
    "render_dashboard_launch_agent",
    "render_dashboard_plist",
    "render_launch_agent",
    "render_plist",
    "start",
    "status",
    "stop",
    "uninstall",
    "uninstall_dashboard",
    "uninstall_dashboard_launch_agent",
    "uninstall_launch_agent",
]
