"""B2.4 phase and accounting controls. Never execute frozen DEV-2 searches."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from usf06.core import Expr, SealedTask, make_tasks
from usf06.known_composition import comparator_digests, protocol_descriptor
from usf06.macro_utility import ARMS, decide_utility, digest
from usf06.macro_validation import run_dev2_validation, run_validation
from usf06.search import search


def training_ast(n):
    v = ["add", ["x"], ["const", n]]
    return ["mul", ["add", v, v], ["sub", v, ["neg", v]]]


class ValidationRunnerTests(unittest.TestCase):
    def test_real_fit_no_macro_keeps_negative_receipt(self):
        fit, validation = make_tasks(78402, 1, 2)
        with patch.object(SealedTask, "holdout_score", side_effect=AssertionError("unexpected score")):
            report = run_validation(fit, validation, ["final-id"], seed=78402, budget=16)
        self.assertEqual(report["decision"]["reason"], "REJECT_NO_MACRO")
        self.assertIsNone(report["candidate"])
        self.assertTrue(report["validation_skipped_no_macro"])
        self.assertEqual(report["validation_rows"], [])
        self.assertEqual(report["validation_scores"], [])
        self.assertEqual(len(report["training"]), 1)
        self.assertEqual(report["fit_totals"]["proposals"], report["training"][0]["proposals"])
        self.assertGreater(report["fit_totals"]["attempts"], 0)
        self.assertEqual(report["cost_ns"]["validation_scoring"], 0)
        self.assertFalse(report["final_search_performed"])
        self.assertFalse(report["milestone_claim_eligible"])

    def test_real_induction_and_validation_from_controlled_fit_outputs(self):
        # Only fit search results are controlled; induction, validation search,
        # validation scoring and utility decision are real repository code.
        fit, validation = make_tasks(78403, 5, 2)
        ids = [t.task_id for t in fit]
        events = []
        original_score = SealedTask.holdout_score

        def instrumented_search(task, budget, seed, mode, **kwargs):
            events.append(("search", task.task_id, seed, kwargs))
            if kwargs.get("train"):
                i = ids.index(task.task_id)
                return SimpleNamespace(
                    proposals=budget, attempts=budget*2,
                    selected=SimpleNamespace(
                        candidate=Expr.from_serial(training_ast(i+1)),
                        survived=i < 4))
            return search(task, budget, seed, mode, **kwargs)

        def instrumented_score(task, candidate):
            events.append(("score", task.task_id))
            self.assertNotIn(task.task_id, ids)
            return original_score(task, candidate)

        with patch("usf06.macro_validation.search", side_effect=instrumented_search), patch.object(
            SealedTask, "holdout_score", instrumented_score
        ):
            report = run_validation(fit, validation, ["final-a", "final-b"],
                                    seed=78403, budget=60,
                                    validation_offsets=(200000, 300000))
        self.assertIsNotNone(report["candidate"])
        self.assertEqual(report["candidate"]["source_ids"], sorted(ids[:4]))
        self.assertEqual(report["surviving_training_observations"], 4)
        self.assertEqual([e[0] for e in events], ["search"]*17 + ["score"]*12)
        expected = [(t.task_id, 78403+offset+i*101, arm)
                    for offset in (200000, 300000)
                    for i, t in enumerate(validation) for arm in ARMS]
        for event, (task_id, seed, arm) in zip(events[5:17], expected):
            self.assertEqual(event[1:3], (task_id, seed))
            self.assertEqual(event[3]["known_composition"], arm == "known")
            self.assertEqual(event[3]["macro"] is not None, arm == "macro")
        self.assertEqual(len(report["validation_rows"]), 12)
        self.assertEqual(len(report["validation_scores"]), 12)
        self.assertEqual(report["manifest"]["comparators"], comparator_digests())
        self.assertEqual(report["manifest"]["architecture_sha256"], digest(report["source_hashes"]))
        decision = decide_utility(report["manifest"], report["candidate"], report["validation_rows"])
        self.assertEqual(report["decision"], decision)
        self.assertNotEqual(decision["reason"], "REJECT_INVALID_EVIDENCE")
        self.assertEqual(report["fit_totals"]["proposals"], 300)
        self.assertEqual(report["fit_totals"]["attempts"], 600)
        for arm in ARMS:
            for k in ("proposals", "attempts", "macro_attempts", "macro_proposals", "search_ns"):
                self.assertEqual(report["validation_work_totals"][arm][k],
                                 sum(row[k] for row in report["validation_rows"] if row["arm"] == arm))
        cost = report["cost_ns"]
        self.assertEqual(cost["validation_scoring"], sum(r["score_ns"] for r in report["validation_scores"]))
        self.assertGreaterEqual(cost["fit"], report["fit_totals"]["search_ns"])
        self.assertGreaterEqual(cost["validation"], cost["validation_scoring"] +
                                sum(t["search_ns"] for t in report["validation_work_totals"].values()))
        self.assertGreaterEqual(cost["total_through_gate"],
                                sum(cost[k] for k in ("setup", "fit", "induction", "validation", "gate")))
        self.assertFalse(report["final_search_performed"])
        self.assertFalse(report["matched_actual_compute_established"])

    def test_invalid_split_and_budget_rejected_before_search(self):
        fit, val = make_tasks(78404, 2, 2)
        cases = [
            (fit, fit, ["final"], 20, (100,)),
            (fit, val, [fit[0].task_id], 20, (100,)),
            (fit, val, [val[0]], 20, (100,)),
            (fit, val, ["final"], True, (100,)),
            (fit, val, ["final"], 20, (100, 100)),
        ]
        with patch("usf06.macro_validation.search", side_effect=AssertionError("unexpected search")) as mock:
            for f, v, final, budget, offsets in cases:
                with self.subTest(budget=budget, offsets=offsets), self.assertRaises(ValueError):
                    run_validation(f, v, final, seed=78404, budget=budget, validation_offsets=offsets)
            mock.assert_not_called()

    def test_dev2_wrapper_configuration_without_executing_search(self):
        with patch("usf06.macro_validation.run_validation", return_value={}) as run:
            report = run_dev2_validation(78201)
        fit, val, final_ids = run.call_args.args
        self.assertEqual((len(fit), len(val), len(final_ids)), (20, 6, 6))
        self.assertEqual(len({t.task_id for t in fit+val} | set(final_ids)), 32)
        self.assertTrue(all(type(i) is str for i in final_ids))
        self.assertEqual(run.call_args.kwargs, {"seed": 78201, "budget": 160})
        self.assertEqual(report["protocol_sha256"], digest(protocol_descriptor()))
        with self.assertRaises(ValueError):
            run_dev2_validation(78124)


if __name__ == "__main__":
    unittest.main()
