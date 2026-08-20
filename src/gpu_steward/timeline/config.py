"""Configuration for the local, read-only GPU timeline collector.

The timeline has its own configuration surface.  It intentionally does not
reuse the queue database configuration: observing a remote host must never
change GPU Steward's scheduling or lease state.  Configuration is plain JSON
and only contains connection and observation metadata; command arguments and
credentials are not accepted as a configuration value.

The public dataclasses are deliberately small so callers can construct a
configuration in tests without touching a user's home directory.  The loader
also accepts a couple of equivalent spellings used by early local prototypes
(``disabled_indices`` and ``disabled_gpu_indices``), while serialisation uses
one canonical spelling.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_SAMPLE_INTERVAL_SECONDS = 60.0
DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS = 300.0
DEFAULT_BACKOFF_SECONDS: Tuple[float, ...] = (60.0, 120.0, 300.0, 600.0)
DEFAULT_TIMELINE_CONFIG_PATH = "~/.gpu-steward/timeline.json"
DEFAULT_TIMELINE_DB_PATH = "~/.gpu-steward/timeline.sqlite3"


class TimelineConfigError(ValueError):
    """Raised when a timeline configuration is unsafe or malformed."""


_HOST_RE = re.compile(r"^[^\x00\r\n\t ]+$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.:@+\-/]+$")


def _as_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TimelineConfigError("{} must be a string".format(field_name))
    text = value.strip()
    if not text:
        raise TimelineConfigError("{} must not be empty".format(field_name))
    if "\x00" in text or any(ord(char) < 32 for char in text):
        raise TimelineConfigError("{} contains a control character".format(field_name))
    return text


def _validate_host(value: Any) -> str:
    host = _as_nonempty_text(value, "host")
    # Passing argv to ssh (rather than a shell) is the safety boundary.  A
    # leading dash would nevertheless be parsed as an ssh option, so reject
    # it explicitly instead of trying to quote it.
    if host.startswith("-") or not _HOST_RE.match(host):
        raise TimelineConfigError("host is not a safe ssh target")
    return host


def _validate_name(value: Any, field_name: str) -> str:
    name = _as_nonempty_text(value, field_name)
    if not _SAFE_NAME_RE.match(name) or name.startswith("-"):
        raise TimelineConfigError("{} contains unsupported characters".format(field_name))
    return name


def _as_positive_float(value: Any, field_name: str, minimum: float = 0.001) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TimelineConfigError("{} must be a number".format(field_name)) from exc
    if result < minimum:
        raise TimelineConfigError("{} must be at least {}".format(field_name, minimum))
    return result


def _as_port(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise TimelineConfigError("port must be an integer") from exc
    if port < 1 or port > 65535:
        raise TimelineConfigError("port must be between 1 and 65535")
    return port


def _normalise_indices(values: Optional[Iterable[Any]]) -> Tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TimelineConfigError("disabled GPU indices must be a list")
    result = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TimelineConfigError("disabled GPU indices must be a list") from exc
    for value in iterator:
        # bool is an int subclass but accepting True as GPU 1 is surprising
        # and makes malformed JSON silently alter observation semantics.
        if isinstance(value, bool):
            raise TimelineConfigError("disabled GPU indices must be integers")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise TimelineConfigError("disabled GPU indices must be integers") from exc
        if index < 0:
            raise TimelineConfigError("disabled GPU indices must be non-negative")
        result.append(index)
    return tuple(sorted(set(result)))


def _host_disabled_values(values: Any, name: Optional[str], host: Optional[str]) -> Any:
    """Resolve either a shared list or a per-host disabled-index mapping."""

    if not isinstance(values, Mapping):
        return values
    for key in (name, host):
        if key is not None and key in values:
            return values[key]
    return ()


def _safe_optional_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TimelineConfigError("project must be a string")
    label = value.strip()
    if not label:
        return None
    # Project aliases are metadata, not arbitrary paths.  This also keeps a
    # config file from becoming a channel for command arguments.
    if "\x00" in label or any(ord(char) < 32 for char in label):
        raise TimelineConfigError("project contains a control character")
    if len(label) > 200:
        raise TimelineConfigError("project is too long")
    return label


@dataclass(frozen=True)
class GPUHostConfig:
    """One remote observation target.

    ``host`` is passed as one argv item to ``ssh``.  ``name`` is the stable
    local/display alias and is what is persisted in timeline samples.
    """

    name: str
    host: str
    port: Optional[int] = None
    disabled_gpu_indices: Tuple[int, ...] = ()
    project: Optional[str] = None
    enabled: bool = True
    timeout_seconds: float = 15.0
    ssh_executable: str = "ssh"
    nvidia_smi_executable: str = "nvidia-smi"
    steward_executable: str = "gpu-steward"

    def __post_init__(self):
        object.__setattr__(self, "name", _validate_name(self.name, "name"))
        object.__setattr__(self, "host", _validate_host(self.host))
        object.__setattr__(self, "port", _as_port(self.port))
        object.__setattr__(
            self, "disabled_gpu_indices", _normalise_indices(self.disabled_gpu_indices)
        )
        object.__setattr__(self, "project", _safe_optional_label(self.project))
        if not isinstance(self.enabled, bool):
            raise TimelineConfigError("enabled must be a boolean")
        object.__setattr__(
            self, "timeout_seconds", _as_positive_float(self.timeout_seconds, "timeout_seconds")
        )
        object.__setattr__(self, "ssh_executable", _validate_name(self.ssh_executable, "ssh_executable"))
        object.__setattr__(
            self,
            "nvidia_smi_executable",
            _validate_name(self.nvidia_smi_executable, "nvidia_smi_executable"),
        )
        object.__setattr__(
            self, "steward_executable", _validate_name(self.steward_executable, "steward_executable")
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        name: Optional[str] = None,
        inherited_disabled: Optional[Iterable[Any]] = None,
    ) -> "GPUHostConfig":
        if not isinstance(value, Mapping):
            raise TimelineConfigError("host configuration must be an object")
        host_name = value.get("name", name)
        if host_name is None:
            host_name = value.get("alias")
        if host_name is None:
            host_name = value.get("host")
        disabled = value.get(
            "disabled_gpu_indices",
            value.get("disabled_indices", value.get("disabled_gpus")),
        )
        if disabled is None:
            disabled = inherited_disabled
        disabled = _host_disabled_values(disabled, host_name, value.get("host"))
        return cls(
            name=_as_nonempty_text(host_name, "name"),
            host=_as_nonempty_text(value.get("host", host_name), "host"),
            port=value.get("port"),
            disabled_gpu_indices=disabled or (),
            project=value.get("project", value.get("project_alias")),
            enabled=value.get("enabled", True),
            timeout_seconds=value.get("timeout_seconds", value.get("timeout", 15.0)),
            ssh_executable=value.get("ssh_executable", "ssh"),
            nvidia_smi_executable=value.get("nvidia_smi_executable", "nvidia-smi"),
            steward_executable=value.get("steward_executable", "gpu-steward"),
        )

    def to_mapping(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "host": self.host,
            "disabled_gpu_indices": list(self.disabled_gpu_indices),
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.port is not None:
            result["port"] = self.port
        if self.project is not None:
            result["project"] = self.project
        if self.ssh_executable != "ssh":
            result["ssh_executable"] = self.ssh_executable
        if self.nvidia_smi_executable != "nvidia-smi":
            result["nvidia_smi_executable"] = self.nvidia_smi_executable
        if self.steward_executable != "gpu-steward":
            result["steward_executable"] = self.steward_executable
        return result

    @property
    def disabled_indices(self) -> Tuple[int, ...]:
        """Compatibility alias for the concise configuration spelling."""

        return self.disabled_gpu_indices


@dataclass(frozen=True)
class TimelineConfig:
    """Top-level timeline configuration.

    ``hosts`` may be empty for an initialised but not-yet-configured local
    timeline.  A collector simply has nothing to sample in that case.
    """

    hosts: Tuple[GPUHostConfig, ...] = ()
    timezone: str = "Asia/Singapore"
    database_path: str = DEFAULT_TIMELINE_DB_PATH
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS
    idle_sample_interval_seconds: float = DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS
    backoff_seconds: Tuple[float, ...] = DEFAULT_BACKOFF_SECONDS
    enabled: bool = True

    def __post_init__(self):
        hosts = tuple(self.hosts)
        names = set()
        for host in hosts:
            if not isinstance(host, GPUHostConfig):
                raise TimelineConfigError("hosts must contain GPUHostConfig values")
            if host.name in names:
                raise TimelineConfigError("duplicate host name: {}".format(host.name))
            names.add(host.name)
        object.__setattr__(self, "hosts", hosts)
        timezone = _as_nonempty_text(self.timezone, "timezone")
        object.__setattr__(self, "timezone", timezone)
        if not isinstance(self.database_path, str) or not self.database_path.strip():
            raise TimelineConfigError("database_path must be a non-empty string")
        object.__setattr__(self, "database_path", os.path.abspath(os.path.expanduser(self.database_path)))
        object.__setattr__(
            self,
            "sample_interval_seconds",
            _as_positive_float(self.sample_interval_seconds, "sample_interval_seconds"),
        )
        idle_interval = _as_positive_float(
            self.idle_sample_interval_seconds, "idle_sample_interval_seconds"
        )
        if idle_interval < self.sample_interval_seconds:
            raise TimelineConfigError(
                "idle_sample_interval_seconds must not be shorter than sample_interval_seconds"
            )
        object.__setattr__(self, "idle_sample_interval_seconds", idle_interval)
        backoff = tuple(_as_positive_float(item, "backoff_seconds") for item in self.backoff_seconds)
        if not backoff:
            raise TimelineConfigError("backoff_seconds must not be empty")
        # Backoff values are deliberately non-decreasing so transient failures
        # cannot make a loop sample more aggressively than its healthy cadence.
        if any(right < left for left, right in zip(backoff, backoff[1:])):
            raise TimelineConfigError("backoff_seconds must be non-decreasing")
        object.__setattr__(self, "backoff_seconds", backoff)
        if not isinstance(self.enabled, bool):
            raise TimelineConfigError("enabled must be a boolean")

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "TimelineConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TimelineConfigError("timeline configuration must be an object")

        inherited_disabled = value.get(
            "disabled_gpu_indices",
            value.get("disabled_indices", value.get("disabled_gpus")),
        )
        raw_hosts = value.get("hosts", value.get("host_configs", ()))
        hosts = []
        if isinstance(raw_hosts, Mapping):
            for name, host_value in raw_hosts.items():
                if isinstance(host_value, str):
                    host_value = {"host": host_value}
                hosts.append(
                    GPUHostConfig.from_mapping(
                        host_value, name=str(name), inherited_disabled=inherited_disabled
                    )
                )
        elif isinstance(raw_hosts, Sequence) and not isinstance(raw_hosts, (str, bytes)):
            for item in raw_hosts:
                hosts.append(
                    GPUHostConfig.from_mapping(item, inherited_disabled=inherited_disabled)
                )
        elif raw_hosts:
            raise TimelineConfigError("hosts must be an object or list")
        elif value.get("host") is not None:
            host_value = value.get("host")
            if isinstance(host_value, str):
                host_value = {"host": host_value}
            hosts.append(
                GPUHostConfig.from_mapping(
                    host_value,
                    name=value.get("name"),
                    inherited_disabled=inherited_disabled,
                )
            )

        sampling = value.get("sampling", {})
        if sampling is None:
            sampling = {}
        if not isinstance(sampling, Mapping):
            raise TimelineConfigError("sampling must be an object")
        interval = value.get(
            "sample_interval_seconds",
            sampling.get("interval_seconds", sampling.get("interval", DEFAULT_SAMPLE_INTERVAL_SECONDS)),
        )
        idle_interval = value.get(
            "idle_sample_interval_seconds",
            sampling.get("idle_interval_seconds", DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS),
        )
        backoff = value.get(
            "backoff_seconds",
            sampling.get("backoff_seconds", DEFAULT_BACKOFF_SECONDS),
        )
        return cls(
            hosts=tuple(hosts),
            timezone=value.get("timezone", "Asia/Singapore"),
            database_path=value.get("database_path", value.get("db_path", DEFAULT_TIMELINE_DB_PATH)),
            sample_interval_seconds=interval,
            idle_sample_interval_seconds=idle_interval,
            backoff_seconds=tuple(backoff),
            enabled=value.get("enabled", True),
        )

    @classmethod
    def from_json(cls, text: str) -> "TimelineConfig":
        try:
            value = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise TimelineConfigError("invalid timeline configuration JSON") from exc
        return cls.from_mapping(value)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "TimelineConfig":
        config_path = os.path.abspath(os.path.expanduser(path or DEFAULT_TIMELINE_CONFIG_PATH))
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                return cls.from_json(handle.read())
        except OSError as exc:
            raise TimelineConfigError("cannot read timeline configuration") from exc

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "hosts": [host.to_mapping() for host in self.hosts],
            "timezone": self.timezone,
            "database_path": self.database_path,
            "sample_interval_seconds": self.sample_interval_seconds,
            "idle_sample_interval_seconds": self.idle_sample_interval_seconds,
            "backoff_seconds": list(self.backoff_seconds),
            "enabled": self.enabled,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    def host(self, name: str) -> GPUHostConfig:
        for item in self.hosts:
            if item.name == name or item.host == name:
                return item
        raise TimelineConfigError("unknown observation host")


def default_config_path() -> str:
    """Return the expanded per-user configuration path."""

    return os.path.abspath(os.path.expanduser(DEFAULT_TIMELINE_CONFIG_PATH))


def default_database_path() -> str:
    """Return the expanded per-user timeline SQLite path."""

    return os.path.abspath(os.path.expanduser(DEFAULT_TIMELINE_DB_PATH))


def load_config(path: Optional[str] = None) -> TimelineConfig:
    """Load a JSON timeline config without exposing file contents on errors."""

    return TimelineConfig.load(path)


# Names used by small integrations and earlier prototypes.
HostConfig = GPUHostConfig
ConfigError = TimelineConfigError
TimelineSettings = TimelineConfig


__all__ = [
    "ConfigError",
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_SAMPLE_INTERVAL_SECONDS",
    "DEFAULT_TIMELINE_CONFIG_PATH",
    "DEFAULT_TIMELINE_DB_PATH",
    "GPUHostConfig",
    "HostConfig",
    "TimelineConfig",
    "TimelineConfigError",
    "TimelineSettings",
    "default_config_path",
    "default_database_path",
    "load_config",
]
