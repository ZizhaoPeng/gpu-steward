"""Token-free, cadence-controlled GPU timeline collection loop.

The collector only coordinates a read-only probe and a local store callback.
There is no model client, prompt handling, transcript reader, or remote write
path in this module. Active GPUs run every 60 seconds by default; a fully
idle/disabled host relaxes to five minutes. A
failed probe emits the probe's host-level ``unknown`` sample and backs off at
1, 2, 5, and 10 minutes (remaining at 10 minutes for later failures).
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS,
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
)
from .gpu import GPUSample, GPUProbe, unknown_sample


class CollectorError(RuntimeError):
    """Raised when a collector cannot be configured safely."""


@dataclass(frozen=True)
class CollectionResult:
    """Evidence from one collection pass."""

    sampled_at: float
    samples: Tuple[GPUSample, ...]
    ok: bool
    next_delay_seconds: float
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return not self.ok

    def as_dict(self):
        return {
            "sampled_at": self.sampled_at,
            "ok": self.ok,
            "next_delay_seconds": self.next_delay_seconds,
            "error": self.error,
            "samples": [sample.as_dict() for sample in self.samples],
        }


class ExponentialBackoff:
    """Bounded 1/2/5/10 minute failure schedule."""

    def __init__(self, delays: Iterable[float] = DEFAULT_BACKOFF_SECONDS):
        values = tuple(float(delay) for delay in delays)
        if not values or any(delay <= 0 for delay in values):
            raise CollectorError("backoff delays must be positive")
        if any(right < left for left, right in zip(values, values[1:])):
            raise CollectorError("backoff delays must be non-decreasing")
        self.delays = values
        self.failure_count = 0

    def reset(self):
        self.failure_count = 0

    def failure_delay(self) -> float:
        position = min(self.failure_count, len(self.delays) - 1)
        delay = self.delays[position]
        self.failure_count += 1
        return delay

    @property
    def current_delay(self) -> float:
        position = min(max(self.failure_count - 1, 0), len(self.delays) - 1)
        return self.delays[position]

    def next_failure_delay(self) -> float:
        """Compatibility spelling; advances the failure counter."""

        return self.failure_delay()


Backoff = ExponentialBackoff
BackoffPolicy = ExponentialBackoff


def _sample_result(value: Any) -> Tuple[List[GPUSample], bool, Optional[str]]:
    """Normalise probe adapters while preserving a fail-closed boundary."""

    if value is None:
        raise CollectorError("probe returned no samples")
    if isinstance(value, GPUSample):
        samples = [value]
    else:
        try:
            samples = list(value)
        except TypeError as exc:
            raise CollectorError("probe returned invalid samples") from exc
    if not all(isinstance(item, GPUSample) for item in samples):
        raise CollectorError("probe returned invalid sample rows")
    # GPUProbe exposes last_ok/last_error so a host-level unknown row remains
    # distinguishable from a legitimate future ``unknown`` hardware state.
    return samples, True, None


class CollectorLoop:
    """Collect one or more hosts into a local store-like sink."""

    def __init__(
        self,
        probes: Any,
        sink: Any,
        sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        idle_sample_interval_seconds: float = DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS,
        backoff_seconds: Iterable[float] = DEFAULT_BACKOFF_SECONDS,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if float(sample_interval_seconds) <= 0:
            raise CollectorError("sample interval must be positive")
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.idle_sample_interval_seconds = float(idle_sample_interval_seconds)
        if self.idle_sample_interval_seconds < self.sample_interval_seconds:
            raise CollectorError("idle interval must not be shorter than sample interval")
        self.clock = clock
        self.sleep = sleep
        self.sink = sink
        self.probes = self._normalise_probes(probes)
        self.backoff = ExponentialBackoff(backoff_seconds)
        self._stop_event = threading.Event()
        self.last_result: Optional[CollectionResult] = None

    @staticmethod
    def _normalise_probes(probes: Any) -> Tuple[Any, ...]:
        if isinstance(probes, (GPUProbe,)):
            return (probes,)
        if isinstance(probes, Mapping):
            result = tuple(probes.values())
        else:
            try:
                result = tuple(probes)
            except TypeError:
                result = (probes,)
        for probe in result:
            if not hasattr(probe, "sample") and not callable(probe):
                raise CollectorError("probes must expose sample()")
        return result

    def stop(self):
        self._stop_event.set()

    def clear_stop(self):
        self._stop_event.clear()

    def _write_sample(self, sample: GPUSample):
        sink = self.sink
        if hasattr(sink, "record_gpu_sample"):
            # TimelineStore exposes scalar keyword fields so it can validate
            # and hash the immutable row.  A tiny embedding sink may instead
            # accept the GPUSample object directly; support both without
            # importing the store (and therefore without coupling the
            # collector to SQLite implementation details).
            payload = {
                "sampled_at": sample.sampled_at,
                "host": sample.host,
                "gpu_index": sample.gpu_index,
                "gpu_uuid_short": sample.gpu_uuid_short,
                "state": sample.state,
                "task_name": sample.task_name or "",
                "attribution": sample.attribution,
                "process_basename": sample.process_basename or "",
                "pid": sample.pid,
            }
            try:
                sink.record_gpu_sample(**payload)
            except TypeError:
                sink.record_gpu_sample(sample)
            return
        if hasattr(sink, "record_sample"):
            sink.record_sample(sample)
            return
        if hasattr(sink, "append"):
            sink.append(sample)
            return
        if callable(sink):
            sink(sample)
            return
        raise CollectorError("sink does not accept GPU samples")

    def _write_samples(self, samples: Sequence[GPUSample]):
        """Persist a probe pass in one transaction when the sink supports it."""

        if hasattr(self.sink, "record_gpu_samples"):
            self.sink.record_gpu_samples(samples)
            return
        for sample in samples:
            self._write_sample(sample)

    def _probe_once(self, probe: Any, sampled_at: float) -> Tuple[List[GPUSample], bool, Optional[str]]:
        try:
            if hasattr(probe, "sample"):
                value = probe.sample(sampled_at=sampled_at)
            else:
                value = probe(sampled_at)
            samples, _, _ = _sample_result(value)
            ok = bool(getattr(probe, "last_ok", True))
            error = None if ok else "probe failed"
            return samples, ok, error
        except Exception as exc:
            # Never copy exception text into timeline rows: a runner exception
            # can contain command arguments or a remote path.
            host = getattr(probe, "host", "unknown")
            return [unknown_sample(host, sampled_at)], False, "probe failed"

    def collect_once(self, sampled_at: Optional[float] = None) -> CollectionResult:
        timestamp = float(self.clock() if sampled_at is None else sampled_at)
        all_samples: List[GPUSample] = []
        ok = True
        error: Optional[str] = None
        for probe in self.probes:
            samples, probe_ok, probe_error = self._probe_once(probe, timestamp)
            if not samples and not probe_ok:
                samples = [unknown_sample(getattr(probe, "host", "unknown"), timestamp)]
            self._write_samples(samples)
            all_samples.extend(samples)
            ok = ok and probe_ok
            if probe_error is not None:
                error = "probe failed"
        if ok:
            self.backoff.reset()
            visible_states = {sample.state for sample in all_samples}
            delay = (
                self.idle_sample_interval_seconds
                if visible_states and visible_states.issubset({"idle", "disabled"})
                else self.sample_interval_seconds
            )
        else:
            delay = self.backoff.failure_delay()
        result = CollectionResult(
            sampled_at=timestamp,
            samples=tuple(all_samples),
            ok=ok,
            next_delay_seconds=delay,
            error=error,
        )
        self.last_result = result
        return result

    def run_forever(
        self,
        stop_event: Optional[Any] = None,
        max_iterations: Optional[int] = None,
    ) -> int:
        """Run until stopped; return the number of completed passes.

        ``max_iterations`` is a deterministic test/embedding escape hatch and
        does not change normal LaunchAgent behaviour when omitted.
        """

        if max_iterations is not None and max_iterations < 0:
            raise CollectorError("max_iterations cannot be negative")
        iterations = 0
        external_event = stop_event
        while not self._stop_event.is_set():
            if external_event is not None and external_event.is_set():
                break
            if max_iterations is not None and iterations >= max_iterations:
                break
            result = self.collect_once()
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            delay = result.next_delay_seconds
            # The injected sleep function keeps tests deterministic.  A
            # caller that needs immediate wake-up can set an event and use a
            # sleep function backed by ``event.wait``; the loop checks both
            # stop flags again before its next pass.
            self.sleep(delay)
        return iterations

    def run_iterations(self, count: int) -> List[CollectionResult]:
        """Run a bounded sequence using the injected sleep function."""

        if count < 0:
            raise CollectorError("count cannot be negative")
        results: List[CollectionResult] = []
        for index in range(count):
            if self._stop_event.is_set():
                break
            result = self.collect_once()
            results.append(result)
            if index + 1 < count:
                self.sleep(result.next_delay_seconds)
        return results

    run = run_forever


TimelineCollector = CollectorLoop
Collector = CollectorLoop
CollectLoop = CollectorLoop


__all__ = [
    "Backoff",
    "BackoffPolicy",
    "Collector",
    "CollectorError",
    "CollectorLoop",
    "CollectLoop",
    "CollectionResult",
    "ExponentialBackoff",
    "TimelineCollector",
]
