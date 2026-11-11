import unittest

from usf06.preflight import evaluation_claim_preflight


def row(task, mode, exact):
    return {"task_id": task, "mode": mode, "holdout_exact": exact}


class EvaluationClaimPreflightTests(unittest.TestCase):
    def test_zero_positive_baseline_fails_closed(self):
        rows = [row(task, mode, 0.0) for task in ("T1", "T2")
                for mode in ("random", "cold_adaptive", "adaptive")]
        result = evaluation_claim_preflight(
            rows,
            expected_test_tasks=2,
            configured_budget_per_task=12,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["positive_baseline_successes"], 0)
        self.assertIn("positive_no_memory_baseline_successes", result["blocking_reasons"])
        self.assertIn("matched_wall_budget_enforced", result["blocking_reasons"])
        self.assertIn("concealed_evaluator", result["blocking_reasons"])

    def test_lost_positive_baseline_is_recorded(self):
        rows = [
            row("T1", "random", 0.0), row("T1", "cold_adaptive", 1.0), row("T1", "adaptive", 0.0),
            row("T2", "random", 1.0), row("T2", "cold_adaptive", 0.0), row("T2", "adaptive", 1.0),
        ]
        result = evaluation_claim_preflight(
            rows,
            expected_test_tasks=2,
            configured_budget_per_task=160,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["lost_positive_baseline_task_ids"], ["T1"])
        self.assertIn("all_positive_baseline_successes_retained", result["blocking_reasons"])

    def test_complete_precommitted_contract_can_pass(self):
        rows = [
            row("T1", "random", 0.0), row("T1", "cold_adaptive", 1.0), row("T1", "adaptive", 1.0),
            row("T2", "random", 1.0), row("T2", "cold_adaptive", 0.0), row("T2", "adaptive", 1.0),
        ]
        result = evaluation_claim_preflight(
            rows,
            expected_test_tasks=2,
            configured_budget_per_task=160,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["blocking_reasons"], [])
        self.assertEqual(result["retained_positive_baseline_successes"], 1)

    def test_incomplete_or_unpaired_arms_fail(self):
        result = evaluation_claim_preflight(
            [row("T1", "random", 1.0), row("T1", "cold_adaptive", 1.0), row("T2", "adaptive", 1.0)],
            expected_test_tasks=1,
            configured_budget_per_task=10,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
        )
        self.assertFalse(result["eligible"])
        self.assertIn("task_ids_paired_across_arms", result["blocking_reasons"])

    def test_duplicate_task_ids_fail(self):
        rows = [row("T1", mode, 1.0) for mode in ("random", "cold_adaptive", "adaptive") for _ in range(2)]
        result = evaluation_claim_preflight(
            rows,
            expected_test_tasks=2,
            configured_budget_per_task=10,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
        )
        self.assertFalse(result["eligible"])
        self.assertIn("unique_task_ids_per_arm", result["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
