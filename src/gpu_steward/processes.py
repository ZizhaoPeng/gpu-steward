"""Linux PID identity helpers used for stale lease recovery."""

import os
import signal
from typing import Optional


def process_start_time(pid: int) -> Optional[int]:
    """Return Linux ``/proc/<pid>/stat`` starttime ticks, if available.

    The process name (field two) may contain spaces or parentheses, so the
    parser finds the final closing parenthesis before splitting fields.  A
    missing ``/proc`` entry means the process has already exited.
    """

    if pid < 1:
        return None
    try:
        with open("/proc/{}/stat".format(pid), "r", encoding="utf-8") as handle:
            text = handle.read().strip()
    except (OSError, IOError):
        return None
    closing = text.rfind(")")
    if closing < 0:
        return None
    fields = text[closing + 2 :].split()
    # fields[0] is field 3 (state); field 22 is index 19.
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def pid_matches(pid: Optional[int], expected_start_time: Optional[int]) -> bool:
    """Check PID liveness and, when available, its recorded start time."""

    if pid is None or pid < 1:
        return False
    current = process_start_time(pid)
    if current is None:
        # On Linux, a missing /proc start time is uncertainty, not proof of
        # liveness.  Fail closed so a PID cannot be mistaken for its successor.
        if os.path.isdir("/proc"):
            return False
        # On non-Linux development hosts no /proc is available.  The runtime
        # still refuses to call an absent PID live, while Linux gets the full
        # PID+start-time protection.
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return expected_start_time is None
    # Linux callers must always carry the recorded start time.  A caller that
    # has no identity token cannot safely act on a live PID.
    return expected_start_time is not None and current == expected_start_time


def terminate_if_matches(pid: int, expected_start_time: Optional[int]) -> bool:
    """Send SIGTERM to the process group when the leader identity matches.

    GPU commands often fork worker processes (`torchrun`, data-loader helpers,
    shell wrappers).  The coordinator launches each managed command in a fresh
    session, so sending SIGTERM to the child's process group is the safest way
    to stop the full managed workload without touching unrelated processes.
    """

    if not pid_matches(pid, expected_start_time):
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        return False
    return True
