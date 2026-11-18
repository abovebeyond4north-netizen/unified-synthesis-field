"""Regression checks for internally inconsistent public evaluator results."""
import unittest
from unittest.mock import patch

from usf06.core import Expr, expr
from usf06.evaluator_client import EvaluatorUnavailable
from usf06.evaluator_supervisor import PublicSupervisor


def exact_candidate():
    return expr("add", expr("mul", Expr.const(2), Expr.var()), Expr.const(1))


class FinalScoreConsistencyTests(unittest.TestCase):
    def test_contradictory_scores_fail_before_aggregate_release(self):
        for invalid in (
            {"exact_success": 1, "exact_fraction": 1.0, "mae": 1.0},
            {"exact_success": 0, "exact_fraction": 0.0, "mae": 0.0},
            {"exact_success": 0, "exact_fraction": 0.5, "mae": 0.0},
        ):
            with self.subTest(invalid=invalid), PublicSupervisor(("first", "second")) as s:
                candidates = {a: exact_candidate() for a in s.arms}
                policies = {a: None for a in s.arms}
                for a in s.arms:
                    task = s.open_arm(a)
                    task.training_samples()
                    s.freeze_arm(a, candidates[a], policies[a])
                real = s._exchange
                def corrupt(arm, message):
                    receipt = real(arm, message)
                    if arm == "second":
                        receipt.update(invalid)
                    return receipt
                with patch.object(s, "_exchange", side_effect=corrupt):
                    with self.assertRaises(EvaluatorUnavailable):
                        s.score_all(candidates, policies)
                self.assertFalse(s._released)
                self.assertTrue(s._closed)
                self.assertTrue(all(t._proc.poll() is not None
                                    for t, _ in s._sessions.values()))

    def test_genuine_exact_and_inexact_scores_still_release(self):
        with PublicSupervisor(("exact", "inexact")) as s:
            candidates = {"exact": exact_candidate(), "inexact": Expr.const(0)}
            policies = {a: None for a in s.arms}
            for a in s.arms:
                task = s.open_arm(a)
                task.training_samples()
                s.freeze_arm(a, candidates[a], policies[a])
            result = s.score_all(candidates, policies)["final_receipts"]
            self.assertEqual(result["exact"]["exact_success"], 1)
            self.assertEqual(result["exact"]["exact_fraction"], 1)
            self.assertEqual(result["exact"]["mae"], 0)
            self.assertEqual(result["inexact"]["exact_success"], 0)
            self.assertLess(result["inexact"]["exact_fraction"], 1)
            self.assertGreater(result["inexact"]["mae"], 0)
            self.assertTrue(s._released)


if __name__ == "__main__":
    unittest.main()
