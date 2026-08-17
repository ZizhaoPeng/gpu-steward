"""Errors raised by the GPU Steward runtime.

The command line interface turns these exceptions into versioned JSON error
objects and a non-zero exit status.  Keeping the error classes small and
standard-library-only makes the same failure boundaries usable by callers
that embed the coordinator.
"""


class StewardError(Exception):
    """Base class for expected, user-actionable runtime errors."""


class InventoryError(StewardError):
    """The GPU inventory could not be trusted."""


class InventoryChangedError(InventoryError):
    """The GPU UUID set changed while a lease was still active."""


class StateError(StewardError):
    """The local queue database is invalid or cannot be updated."""


class QueueError(StewardError):
    """A task could not be queued, claimed, or released safely."""


class TaskNotFoundError(QueueError):
    """The requested task does not exist in this user's queue."""


class TaskBusyError(QueueError):
    """A task is already owned by another supervisor."""


class CommandError(StewardError):
    """The requested command is invalid or failed before task execution."""
