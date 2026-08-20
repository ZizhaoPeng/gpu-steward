#!/usr/bin/env python3
"""Forward one official Codex hook payload to the local GPU Steward CLI.

This process is intentionally a small, token-free stdin bridge. It does not
parse, persist, print, or summarize the payload; the timeline CLI owns schema
validation and sanitization. ``exec`` replaces this bridge process instead of
spawning a second Python process, and the hook host owns the short timeout.
"""

from __future__ import annotations

import os
import shutil
import sys


def main() -> int:
    executable = (
        os.environ.get("GPU_STEWARD_EXECUTABLE")
        or shutil.which("gpu-steward")
        or os.path.expanduser("~/.gpu-steward/venv/bin/gpu-steward")
    )
    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        return 0
    # Hook collection is best-effort observability. Suppress all output so a
    # local storage problem can never add text to the model context.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    if devnull > sys.stderr.fileno():
        os.close(devnull)
    try:
        os.execv(executable, [executable, "timeline", "hook"])
    except OSError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
