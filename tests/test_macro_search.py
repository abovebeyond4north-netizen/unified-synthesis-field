"""Controls for experimental macro reuse in the real USF search."""
import unittest
from unittest.mock import patch

from usf06.core import SealedTask, make_tasks, sha256
from usf06.macro_induction import induce_root_macro
from usf06.macro_trial import run_macro_trial
from usf06.search import search


def fixture_macro():
    # Positive mechanism control only; not a learned trial artifact.
    return induce_root_macro([
        (str(i), ("mul", ("add", ("x",), ("const", i)),
                  ("add", ("x",), ("const", i))))
        for i in range(1, 5)
    ])


class MacroSearchTests(unittest.TestCase):
    def test_opt_in_reuses_macro_under_original_caps(self):
        task = make_tasks(78123, 2, 2)[1][0]
        macro = fixture_macro()
        self.assertIsNotNone(macro)
        plain = search(task, 80, 1234, "evolution")
        self.assertEqual(plain, search(task, 80, 1234, "evolution", macro=None))
        extended = search(task, 80, 1234, "evolution", macro=macro)
        self.assertEqual(extended, search(task, 80, 1234, "evolution", macro=macro))
        self.assertEqual((plain.macro_attempts, plain.macro_proposals), (0, 0))
        self.assertGreater(extended.macro_proposals, 0)
        self.assertLessEqual(extended.macro_proposals, extended.macro_attempts)
        self.assertLessEqual(extended.macro_attempts, extended.attempts)
        for result in (plain, extended):
            self.assertLessEqual(result.proposals, 80)
            self.assertLessEqual(result.attempts, 800)
            self.assertLessEqual(result.selected.candidate.size(), 23)
        with self.assertRaises(ValueError):
            search(task, 80, 1234, "random", macro=macro)
        with self.assertRaises(ValueError):
            search(task, 80, 1234, "evolution", train=True, macro=macro)

    def test_real_training_freezes_before_all_test_scores(self):
        train, test = make_tasks(78123, 6, 2)
        calls = []
        real_score = SealedTask.holdout_score

        def track_search(task, *args, **kwargs):
            calls.append(("search", task.task_id, kwargs.get("train", False)))
            return search(task, *args, **kwargs)

        def track_score(task, candidate):
            calls.append(("score", task.task_id))
            return real_score(task, candidate)

        with patch("usf06.macro_trial.search", side_effect=track_search), patch.object(
            SealedTask, "holdout_score", track_score
        ):
            report = run_macro_trial(train, test, seed=78123, budget=36)

        expected = ([("search", t.task_id, True) for t in train]
                    + [("search", t.task_id, False) for t in test for _ in range(2)]
                    + [("score", t.task_id) for t in test for _ in range(2)])
        self.assertEqual(calls, expected)
        self.assertEqual(report["macro_commitment"], sha256(report["frozen_macro"]))
        self.assertEqual(len(report["test_rows"]), 4)
        for row in report["test_rows"]:
            self.assertLessEqual(row["proposals"], 36)
            self.assertLessEqual(row["attempts"], 360)
        self.assertFalse(report["milestone_claim_eligible"])
        self.assertFalse(report["matched_actual_compute_established"])
        self.assertFalse(report["independent_provenance_authenticated"])

    def test_no_induced_macro_is_preserved_and_split_overlap_rejected(self):
        train, test = make_tasks(78123, 1, 2)
        report = run_macro_trial(train, test, seed=78123, budget=12)
        self.assertEqual(report["status"], "NO_MACRO_INDUCED")
        self.assertIsNone(report["frozen_macro"])
        self.assertEqual(report["paired_exact_delta"], 0)
        self.assertEqual(report["arm_totals"]["macro"]["macro_attempts"], 0)
        for field in ("exact_success", "proposals", "attempts"):
            self.assertEqual(report["arm_totals"]["macro"][field],
                             report["arm_totals"]["primitive"][field])
        with self.assertRaisesRegex(ValueError, "unique and disjoint"):
            run_macro_trial(train, train, seed=78123, budget=12)


if __name__ == "__main__":
    unittest.main()
