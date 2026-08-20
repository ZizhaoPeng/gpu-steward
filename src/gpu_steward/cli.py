"""Command line interface for GPU Steward."""

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from .errors import CommandError, StewardError
from .runtime import Coordinator
from .state import SCHEMA_VERSION, StateStore
from .timeline.cli import TimelineCLIError, add_parser as add_timeline_parser, dispatch as dispatch_timeline
from .timeline.config import TimelineConfigError
from .timeline.launchd import LaunchAgentError
from .timeline.store import TimelineError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-steward",
        description="Queue whole NVIDIA GPUs for concurrent SSH/Codex sessions.",
    )
    parser.add_argument("--db", dest="db", default=None, help="private SQLite state path")
    parser.add_argument("--reserve", type=int, default=1, help="solo-job GPU reserve")
    parser.add_argument(
        "--strict-fifo", action="store_true", help="block the queue behind an infeasible head task"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("doctor", "inventory", "status", "gc"):
        command = sub.add_parser(name)
        command.add_argument("--db", dest="sub_db", default=None)
        command.add_argument("--json", action="store_true", help="emit versioned JSON")

    cancel = sub.add_parser("cancel")
    cancel.add_argument("task_id")
    cancel.add_argument("--db", dest="sub_db", default=None)
    cancel.add_argument("--json", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--db", dest="sub_db", default=None)
    run.add_argument("--json", action="store_true", help="write result JSON to stderr")
    run.add_argument("--min", "--min-gpus", dest="min_gpus", type=int, default=1)
    run.add_argument("--max", "--max-gpus", dest="max_gpus", default="auto")
    run.add_argument("--priority", type=int, default=0)
    run.add_argument("--label", default="")
    run.add_argument("--cwd", default=None)
    run.add_argument("--wait-timeout", type=float, default=None)
    run.add_argument("command_argv", nargs=argparse.REMAINDER)
    add_timeline_parser(sub)
    return parser


def _parse_max(value: str):
    if value is None or str(value).lower() == "auto":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CommandError("--max must be a positive integer or auto") from exc
    if parsed < 1:
        raise CommandError("--max must be at least 1")
    return parsed


def _emit(payload, stream):
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def _error_payload(exc: Exception):
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "error": {"type": exc.__class__.__name__, "message": str(exc)},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "timeline":
        try:
            return dispatch_timeline(args)
        except (TimelineCLIError, TimelineConfigError, TimelineError, LaunchAgentError) as exc:
            _emit(_error_payload(exc), sys.stderr)
            return 2
        except KeyboardInterrupt:
            _emit(_error_payload(CommandError("interrupted")), sys.stderr)
            return 130
    if args.reserve < 0:
        parser.error("--reserve cannot be negative")
    db_path = getattr(args, "sub_db", None) or args.db
    store = None
    coordinator = None
    try:
        store = StateStore(db_path)
        coordinator = Coordinator(
            store=store,
            reserve_gpus=args.reserve,
            strict_fifo=args.strict_fifo,
        )
        if args.command == "doctor":
            payload = coordinator.doctor_payload()
            _emit(payload, sys.stdout)
            return 0 if payload["ok"] else 1
        if args.command == "inventory":
            _emit(coordinator.inventory_payload(), sys.stdout)
            return 0
        if args.command == "status":
            _emit(coordinator.status_payload(), sys.stdout)
            return 0
        if args.command == "gc":
            _emit(coordinator.gc(), sys.stdout)
            return 0
        if args.command == "cancel":
            payload = coordinator.cancel_task(args.task_id)
            _emit(payload, sys.stdout)
            return 0 if payload["ok"] else 1
        if args.command == "run":
            command = list(args.command_argv)
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                raise CommandError("run requires an argv after --")
            max_gpus = _parse_max(args.max_gpus)
            exit_code, payload = coordinator.run_task(
                command=command,
                cwd=args.cwd,
                min_gpus=args.min_gpus,
                max_gpus=max_gpus,
                priority=args.priority,
                label=args.label,
                wait_timeout=args.wait_timeout,
            )
            # Child stdout/stderr remain untouched.  The machine-readable
            # supervisor result goes to stderr so a wrapped command may use
            # stdout for its own protocol.
            _emit(payload, sys.stderr)
            # Popen uses negative values for signal exits, while shell exit
            # statuses use the conventional 128 + signal number mapping.
            return exit_code if exit_code >= 0 else 128 + abs(exit_code)
        raise CommandError("unknown command: {}".format(args.command))
    except StewardError as exc:
        _emit(_error_payload(exc), sys.stderr)
        return 2
    except KeyboardInterrupt:
        _emit(
            _error_payload(CommandError("interrupted")),
            sys.stderr,
        )
        return 130
    finally:
        if coordinator is not None:
            coordinator.close()
        elif store is not None:
            store.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
