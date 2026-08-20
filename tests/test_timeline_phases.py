import unittest

from gpu_steward.errors import CommandError
from gpu_steward.timeline.phases import CODEX_PHASES, normalize_phase


class TimelinePhaseTests(unittest.TestCase):
    def test_normalizes_only_frozen_phase_vocabulary(self):
        self.assertEqual("waiting-tool", normalize_phase(" Waiting_Tool "))
        self.assertIn("research", CODEX_PHASES)
        self.assertIn("suspected-stall", CODEX_PHASES)

    def test_unknown_phase_fails_closed(self):
        with self.assertRaises(CommandError):
            normalize_phase("thinking-maybe")


if __name__ == "__main__":
    unittest.main()
