"""Fixed public development trial; no concealed evaluation or promotion.

Run: python -m unittest discover -s tests -p test_macro_public_trial.py -v
The JSON receipts are written to hosted CI logs even for zero/negative gains.
The statistical unit is a seed, not a Python-version repetition.
"""
import json
import unittest

from usf06.core import make_tasks, sha256
from usf06.macro_trial import run_macro_trial


SEEDS = (78124, 78125, 78126)
BUDGET = 160
TRAIN_TASKS = 20
TEST_TASKS = 6
PROTOCOL = "USF-MACRO-PUBLIC-DEV-1"


class PublicMacroTrialTests(unittest.TestCase):
    def test_frozen_public_development_comparison(self):
        receipts = []
        for seed in SEEDS:
            training, test = make_tasks(seed, TRAIN_TASKS, TEST_TASKS)
            report = run_macro_trial(training, test, seed=seed, budget=BUDGET)
            print("USF_MACRO_RECEIPT " + json.dumps(
                {"protocol": PROTOCOL, **report}, sort_keys=True), flush=True)
            receipts.append(report)
            self.assertEqual(report["macro_commitment"], sha256(report["frozen_macro"]))
            self.assertFalse(report["milestone_claim_eligible"])
            self.assertFalse(report["matched_actual_compute_established"])
            self.assertFalse(report["formal_verification_performed"])
            self.assertEqual(len(report["training"]), TRAIN_TASKS)
            self.assertEqual(len(report["test_rows"]), 2 * TEST_TASKS)
            for row in report["test_rows"]:
                self.assertLessEqual(row["proposals"], BUDGET)
                self.assertLessEqual(row["attempts"], BUDGET * 10)
                self.assertLessEqual(row["macro_proposals"], row["macro_attempts"])
                self.assertLessEqual(row["macro_attempts"], row["attempts"])
            if report["status"] == "NO_MACRO_INDUCED":
                self.assertIsNone(report["frozen_macro"])
                self.assertEqual(report["paired_exact_delta"], 0)
        totals = {
            arm: {field: sum(r["arm_totals"][arm][field] for r in receipts)
                  for field in ("exact_success", "proposals", "attempts",
                                "macro_attempts", "macro_proposals", "search_ns",
                                "holdout_scoring_ns")}
            for arm in ("primitive", "macro")
        }
        summary = {
            "protocol": PROTOCOL, "seeds": SEEDS, "train_tasks_per_seed": TRAIN_TASKS,
            "test_tasks_per_seed": TEST_TASKS, "candidate_cap": BUDGET,
            "paired_tasks": len(SEEDS) * TEST_TASKS,
            "induced_runs": sum(r["frozen_macro"] is not None for r in receipts),
            "no_macro_runs": sum(r["frozen_macro"] is None for r in receipts),
            "surviving_observations_by_seed": [r["surviving_training_observations"] for r in receipts],
            "per_seed_delta": [r["paired_exact_delta"] for r in receipts],
            "arm_totals": totals,
            "net_exact_successes": totals["macro"]["exact_success"] - totals["primitive"]["exact_success"],
            "shared_training_ns": sum(r["shared_training_ns"] for r in receipts),
            "induction_ns": sum(r["induction_ns"] for r in receipts),
            "scientific_acceptance": False,
        }
        print("USF_MACRO_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        # A negative experiment is not a failed software test.
        self.assertEqual(summary["induced_runs"] + summary["no_macro_runs"], len(SEEDS))


if __name__ == "__main__":
    unittest.main()
