import json
from pathlib import Path
import unittest

from usf06.preflight import (
    REQUIRED_ARMS,
    REQUIRED_PROVENANCE_RECEIPTS,
    evaluation_claim_preflight,
)


def row(task, mode, exact):
    return {"task_id": task, "mode": mode, "holdout_exact": exact}


def preflight(rows, **kwargs):
    """Supply the frozen adaptive-arm order as the external order receipt."""
    expected_task_order = list(dict.fromkeys(
        item["task_id"] for item in rows if item.get("mode") == "adaptive"
    ))
    return evaluation_claim_preflight(
        rows,
        expected_task_order=expected_task_order,
        **kwargs,
    )


MODES = ("random", "evolution", "cold_adaptive", "adaptive")


class EvaluationClaimPreflightTests(unittest.TestCase):
    def test_zero_positive_baseline_fails_closed(self):
        rows = [row(task, mode, 0.0) for task in ("T1", "T2")
                for mode in MODES]
        result = preflight(
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
            row("T1", "random", 0.0), row("T1", "evolution", 0.0),
            row("T1", "cold_adaptive", 1.0), row("T1", "adaptive", 0.0),
            row("T2", "random", 1.0), row("T2", "evolution", 1.0),
            row("T2", "cold_adaptive", 0.0), row("T2", "adaptive", 1.0),
        ]
        result = preflight(
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
            row("T1", "random", 0.0), row("T1", "evolution", 0.0),
            row("T1", "cold_adaptive", 1.0), row("T1", "adaptive", 1.0),
            row("T2", "random", 1.0), row("T2", "evolution", 1.0),
            row("T2", "cold_adaptive", 0.0), row("T2", "adaptive", 1.0),
        ]
        result = preflight(
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
        self.assertFalse(result["claim_accepted"])
        self.assertIn("bootstrap_95pct_lower_bound_exceeds_zero", result["claim_blocking_reasons"])

    def test_forged_confidence_bound_cannot_accept_claim(self):
        rows = [
            row("T1", "random", 0.0), row("T1", "evolution", 0.0),
            row("T1", "cold_adaptive", 0.0), row("T1", "adaptive", 1.0),
            row("T2", "random", 1.0), row("T2", "evolution", 1.0),
            row("T2", "cold_adaptive", 1.0), row("T2", "adaptive", 1.0),
        ]
        result = preflight(
            rows,
            expected_test_tasks=2,
            configured_budget_per_task=160,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
            bootstrap_seed=1234,
            paired_advantage_ci_lower_bound=0.9,
        )
        self.assertTrue(result["eligible"])
        self.assertFalse(result["claim_accepted"])
        self.assertEqual(result["paired_mean_difference"], 0.5)
        self.assertEqual(result["paired_advantage_bootstrap_95pct_CI"], [0.0, 1.0])
        self.assertFalse(result["claim_checks"]["reported_bootstrap_lower_bound_consistent"])

    def test_recomputed_positive_bound_can_accept_claim(self):
        rows = []
        for i in range(20):
            task = f"T{i:02d}"
            baseline = 1.0 if i == 0 else 0.0
            rows.extend((
                row(task, "random", 0.0),
                row(task, "evolution", 0.0),
                row(task, "cold_adaptive", baseline),
                row(task, "adaptive", 1.0),
            ))
        result = preflight(
            rows,
            expected_test_tasks=20,
            configured_budget_per_task=160,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
            bootstrap_seed=1234,
        )
        self.assertTrue(result["eligible"])
        self.assertTrue(result["claim_accepted"])
        self.assertFalse(result["milestone_a_accepted"])
        self.assertIn(
            "independently_authenticated_provenance",
            result["milestone_a_blocking_reasons"],
        )
        self.assertEqual(result["paired_mean_difference"], 0.95)
        self.assertGreater(result["paired_advantage_ci_lower_bound"], 0.0)
        replay = preflight(
            rows,
            expected_test_tasks=20,
            configured_budget_per_task=160,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
            bootstrap_seed=1234,
        )
        self.assertEqual(
            result["paired_advantage_bootstrap_95pct_CI"],
            replay["paired_advantage_bootstrap_95pct_CI"],
        )

    def test_matching_caller_supplied_receipts_do_not_authenticate_milestone_a(self):
        rows = []
        for i in range(20):
            task = f"T{i:02d}"
            baseline = 1.0 if i == 0 else 0.0
            rows.extend((
                row(task, "random", 0.0),
                row(task, "evolution", 0.0),
                row(task, "cold_adaptive", baseline),
                row(task, "adaptive", 1.0),
            ))
        receipts = {
            name: f"{index + 1:064x}"
            for index, name in enumerate(REQUIRED_PROVENANCE_RECEIPTS)
        }
        result = preflight(
            rows,
            expected_test_tasks=20,
            configured_budget_per_task=160,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
            bootstrap_seed=1234,
            provenance_receipts=receipts,
            expected_provenance_receipts=dict(receipts),
        )
        self.assertTrue(result["claim_accepted"])
        self.assertFalse(result["milestone_a_accepted"])
        self.assertTrue(result["receipt_strings_match"])
        self.assertIn(
            "independently_authenticated_provenance",
            result["milestone_a_blocking_reasons"],
        )
        self.assertTrue(all(result["provenance_checks"].values()))

    def test_missing_malformed_incomplete_or_mismatched_provenance_fails_closed(self):
        rows = []
        for i in range(20):
            task = f"T{i:02d}"
            baseline = 1.0 if i == 0 else 0.0
            rows.extend((
                row(task, "random", 0.0),
                row(task, "evolution", 0.0),
                row(task, "cold_adaptive", baseline),
                row(task, "adaptive", 1.0),
            ))
        expected = {
            name: f"{index + 1:064x}"
            for index, name in enumerate(REQUIRED_PROVENANCE_RECEIPTS)
        }
        malformed = dict(expected)
        malformed["protocol_sha256"] = "not-a-sha256"
        incomplete = dict(expected)
        incomplete.pop("target_commitment")
        mismatched = dict(expected)
        mismatched["evaluator_sha256"] = "f" * 64
        cases = (
            None,
            "serialized-but-unauthenticated",
            malformed,
            incomplete,
            mismatched,
        )
        for receipts in cases:
            with self.subTest(receipts=receipts):
                result = preflight(
                    rows,
                    expected_test_tasks=20,
                    configured_budget_per_task=160,
                    concealed_evaluator=True,
                    independent_administration=True,
                    matched_wall_budget_enforced=True,
                    bootstrap_seed=1234,
                    provenance_receipts=receipts,
                    expected_provenance_receipts=expected,
                )
                self.assertTrue(result["claim_accepted"])
                self.assertFalse(result["milestone_a_accepted"])
                self.assertIn(
                    "independently_authenticated_provenance",
                    result["milestone_a_blocking_reasons"],
                )

    def test_missing_or_invalid_bootstrap_configuration_fails_closed(self):
        rows = []
        for i in range(20):
            task = f"T{i:02d}"
            baseline = 1.0 if i == 0 else 0.0
            rows.extend((
                row(task, "random", 0.0),
                row(task, "evolution", 0.0),
                row(task, "cold_adaptive", baseline),
                row(task, "adaptive", 1.0),
            ))
        for seed, iterations in ((None, 2000), (True, 2000), ("1234", 2000), (1234, 1999), (1234, True)):
            with self.subTest(seed=seed, iterations=iterations):
                result = preflight(
                    rows,
                    expected_test_tasks=20,
                    configured_budget_per_task=160,
                    concealed_evaluator=True,
                    independent_administration=True,
                    matched_wall_budget_enforced=True,
                    bootstrap_seed=seed,
                    bootstrap_iterations=iterations,
                )
                self.assertTrue(result["eligible"])
                self.assertFalse(result["claim_accepted"])
                self.assertFalse(result["claim_checks"]["recorded_bootstrap_configuration"])
                self.assertEqual(result["paired_advantage_bootstrap_95pct_CI"], [None, None])


    def test_configuration_evidence_requires_literal_true(self):
        # Python bool('false') evaluates True; malformed serialized flags
        # must never establish evaluator independence, concealment, or budget matching.
        rows = [
            row("T1", "random", 0.0), row("T1", "evolution", 0.0),
            row("T1", "cold_adaptive", 0.0), row("T1", "adaptive", 1.0),
            row("T2", "random", 1.0), row("T2", "evolution", 1.0),
            row("T2", "cold_adaptive", 1.0), row("T2", "adaptive", 1.0),
        ]
        for malformed in ("false", "true", 1, [], {"verified": True}, None):
            with self.subTest(malformed=malformed):
                for flag in (
                    "matched_wall_budget_enforced",
                    "concealed_evaluator",
                    "independent_administration",
                ):
                    options = {
                        "matched_wall_budget_enforced": True,
                        "concealed_evaluator": True,
                        "independent_administration": True,
                    }
                    options[flag] = malformed
                    result = preflight(
                        rows,
                        expected_test_tasks=2,
                        configured_budget_per_task=160,
                        bootstrap_seed=1234,
                        **options,
                    )
                    self.assertFalse(result["eligible"])
                    self.assertFalse(result["claim_accepted"])
                    self.assertIn(flag, result["blocking_reasons"])
                    self.assertIn("preliminary_eligibility", result["claim_blocking_reasons"])

    def test_incomplete_or_unpaired_arms_fail(self):
        result = preflight(
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
        rows = [row("T1", mode, 1.0) for mode in MODES for _ in range(2)]
        result = preflight(
            rows,
            expected_test_tasks=2,
            configured_budget_per_task=10,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
        )
        self.assertFalse(result["eligible"])
        self.assertIn("unique_task_ids_per_arm", result["blocking_reasons"])

    def test_missing_null_malformed_nonfinite_and_out_of_range_outcomes_fail(self):
        invalid = (None, "1.0", float("nan"), float("inf"), -0.1, 1.1)
        for value in invalid:
            with self.subTest(value=value):
                rows = [
                    row(task, mode, 1.0)
                    for task in ("T1", "T2")
                    for mode in MODES
                ]
                rows[-1]["holdout_exact"] = value
                result = preflight(
                    rows,
                    expected_test_tasks=2,
                    configured_budget_per_task=10,
                    concealed_evaluator=True,
                    independent_administration=True,
                    matched_wall_budget_enforced=True,
                    paired_advantage_ci_lower_bound=0.1,
                )
                self.assertFalse(result["eligible"])
                self.assertIn("valid_correctness_outcomes", result["blocking_reasons"])

    def test_missing_evolutionary_comparator_fails_closed(self):
        rows = []
        for i in range(20):
            task = f"T{i:02d}"
            baseline = 1.0 if i == 0 else 0.0
            rows.extend((
                row(task, "random", 0.0),
                row(task, "cold_adaptive", baseline),
                row(task, "adaptive", 1.0),
            ))
        result = preflight(
            rows,
            expected_test_tasks=20,
            configured_budget_per_task=160,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
            bootstrap_seed=1234,
        )
        self.assertFalse(result["eligible"])
        self.assertFalse(result["claim_accepted"])
        self.assertFalse(result["checks"]["complete_required_arms"])
        self.assertEqual(result["observed_rows_by_arm"]["evolution"], 0)
        self.assertIn("evolution", result["required_arms"])

    def test_reordered_arm_fails_frozen_order_receipt(self):
        rows = [
            row(task, mode, 1.0)
            for task in ("T1", "T2")
            for mode in MODES
        ]
        evolution = [item for item in rows if item["mode"] == "evolution"]
        rows = [item for item in rows if item["mode"] != "evolution"] + list(reversed(evolution))
        result = preflight(
            rows,
            expected_test_tasks=2,
            configured_budget_per_task=10,
            concealed_evaluator=True,
            independent_administration=True,
            matched_wall_budget_enforced=True,
        )
        self.assertTrue(result["checks"]["task_ids_paired_across_arms"])
        self.assertFalse(result["eligible"])
        self.assertIn("task_order_matches_frozen_receipt", result["blocking_reasons"])
        self.assertEqual(result["observed_task_order_by_arm"]["evolution"], ["T2", "T1"])

    def test_missing_frozen_order_receipt_fails_closed(self):
        rows = [row(task, mode, 1.0) for task in ("T1", "T2") for mode in MODES]
        for receipt in (None, "T1,T2", 1, ["T1", {"task": "T2"}]):
            with self.subTest(receipt=receipt):
                result = evaluation_claim_preflight(
                    rows,
                    expected_test_tasks=2,
                    configured_budget_per_task=10,
                    expected_task_order=receipt,
                    concealed_evaluator=True,
                    independent_administration=True,
                    matched_wall_budget_enforced=True,
                )
                self.assertFalse(result["eligible"])
                self.assertIn("task_order_matches_frozen_receipt", result["blocking_reasons"])

    def test_preregistered_protocol_freezes_all_required_arms(self):
        protocol_path = Path(__file__).resolve().parents[1] / "preregistered_concealed_evaluation_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [arm["name"] for arm in protocol["frozen_arms"]],
            list(REQUIRED_ARMS),
        )
        self.assertEqual(
            protocol["receipts"][:4],
            list(REQUIRED_PROVENANCE_RECEIPTS),
        )
        self.assertIn("missing evolutionary comparator rows", protocol["automatic_blockers"])
        self.assertIn(
            "missing or mismatched per-arm task-order receipt",
            protocol["automatic_blockers"],
        )


if __name__ == "__main__":
    unittest.main()
