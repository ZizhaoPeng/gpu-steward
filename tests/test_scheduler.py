import unittest

from gpu_steward.scheduler import Request, plan_allocations


GPUS = ("GPU-0", "GPU-1", "GPU-2", "GPU-3")


class SchedulerTests(unittest.TestCase):
    def test_first_job_gets_three_of_four(self):
        result = plan_allocations(
            free_gpu_ids=GPUS,
            waiting=[Request("first")],
            total_gpus=4,
        )
        self.assertEqual([("first", GPUS[:3])], self._pairs(result))

    def test_second_job_gets_reserved_gpu(self):
        result = plan_allocations(
            free_gpu_ids=("GPU-3",),
            waiting=[Request("second")],
            total_gpus=4,
            active_jobs=1,
        )
        self.assertEqual([("second", ("GPU-3",))], self._pairs(result))

    def test_three_released_gpus_split_two_plus_one(self):
        result = plan_allocations(
            free_gpu_ids=GPUS[:3],
            waiting=[Request("older", created_at=1), Request("newer", created_at=2)],
            total_gpus=4,
            active_jobs=1,
        )
        self.assertEqual(
            [("older", ("GPU-0", "GPU-1")), ("newer", ("GPU-2",))],
            self._pairs(result),
        )

    def test_one_released_gpu_starts_only_oldest(self):
        result = plan_allocations(
            free_gpu_ids=("GPU-2",),
            waiting=[Request("older", created_at=1), Request("newer", created_at=2)],
            total_gpus=4,
            active_jobs=2,
        )
        self.assertEqual([("older", ("GPU-2",))], self._pairs(result))

    def test_two_waiters_split_empty_four_evenly(self):
        result = plan_allocations(
            free_gpu_ids=GPUS,
            waiting=[Request("a", created_at=1), Request("b", created_at=2)],
            total_gpus=4,
        )
        self.assertEqual(
            [("a", ("GPU-0", "GPU-1")), ("b", ("GPU-2", "GPU-3"))],
            self._pairs(result),
        )

    def test_external_busy_host_does_not_hold_extra_reserve(self):
        result = plan_allocations(
            free_gpu_ids=("GPU-0", "GPU-2"),
            waiting=[Request("managed")],
            total_gpus=4,
            external_busy_gpus=2,
        )
        self.assertEqual(
            [("managed", ("GPU-0", "GPU-2"))], self._pairs(result)
        )

    def test_backfills_around_infeasible_large_request(self):
        result = plan_allocations(
            free_gpu_ids=("GPU-0",),
            waiting=[
                Request("needs-two", min_gpus=2, created_at=1),
                Request("needs-one", created_at=2),
            ],
            total_gpus=4,
            active_jobs=1,
        )
        self.assertEqual([("needs-one", ("GPU-0",))], self._pairs(result))

    def test_respects_maximum(self):
        result = plan_allocations(
            free_gpu_ids=GPUS,
            waiting=[Request("one-only", max_gpus=1)],
            total_gpus=4,
        )
        self.assertEqual([("one-only", ("GPU-0",))], self._pairs(result))

    @staticmethod
    def _pairs(allocations):
        return [(item.request_id, item.gpu_ids) for item in allocations]


if __name__ == "__main__":
    unittest.main()
