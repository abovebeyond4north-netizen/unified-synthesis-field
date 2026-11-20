"""Public subprocess controls for exact request-byte reconciliation."""
import unittest
from unittest.mock import patch

from usf06.core import Expr, sha256
from usf06.evaluator_client import EvaluatorUnavailable
from usf06.evaluator_supervisor import PublicSupervisor


class RequestByteReconciliationTests(unittest.TestCase):
    def test_forged_byte_counts_fail_despite_valid_receipt_hash(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta), PublicSupervisor(("one",)) as s:
                t = s.open_arm("one")
                t.training_samples()
                candidate = Expr.const(0)
                t.falsify(candidate)
                real = s._exchange

                def forged(arm, message):
                    response = real(arm, message)
                    f = response["receipt"]
                    f["learner"]["request_bytes"] += delta
                    f["freeze_sha256"] = sha256({
                        k: v for k, v in f.items() if k != "freeze_sha256"
                    })
                    return response

                with patch.object(s, "_exchange", side_effect=forged):
                    with self.assertRaises(EvaluatorUnavailable):
                        s.freeze_arm("one", candidate, None)
                self.assertTrue(s._closed)
                self.assertFalse(s._released)
                self.assertNotIn("one", s._frozen)
                self.assertIsNotNone(t._proc.poll())

    def test_actual_bytes_reconcile_and_cached_observe_sends_nothing(self):
        import os
        with PublicSupervisor(("one",)) as s:
            t = s.open_arm("one")
            candidate = Expr.const(0)
            fd = t._proc.stdin.fileno()
            real_write = os.write
            counts = []

            def short_write(target, data):
                # Force several short writes per frame, counting actual bytes.
                if target == fd:
                    n = real_write(target, data[:17])
                    counts.append(n)
                    return n
                return real_write(target, data)

            with patch("usf06.evaluator_client.os.write", side_effect=short_write):
                t.training_samples()
                observed = sum(counts)
                t.training_samples()
                self.assertEqual(sum(counts), observed)
                t.falsify(candidate)
                t.falsify(candidate)
            self.assertGreater(len(counts), 3)
            receipt = s.freeze_arm("one", candidate, None)
            self.assertEqual(receipt["learner"]["frames"], 3)
            self.assertEqual(receipt["learner"]["request_bytes"], sum(counts))
            self.assertGreater(sum(counts), observed)
            result = s.score_all({"one": candidate}, {"one": None})
            self.assertFalse(result["matched_actual_compute_established"])
            self.assertTrue(s._released)


if __name__ == "__main__":
    unittest.main()
