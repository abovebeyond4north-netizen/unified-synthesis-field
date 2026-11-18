"""Public synthetic supervisor tests using real worker processes."""
import json
import unittest
from unittest.mock import patch

from usf06.core import Expr, expr, sha256
from usf06.evaluator_client import EvaluatorUnavailable
from usf06.evaluator_supervisor import PublicSupervisor, DEFAULT_ARMS
from usf06.search import AdaptivePolicy, search


def known():
    return expr("add",expr("mul",Expr.const(2),Expr.var()),Expr.const(1))


class SupervisorTests(unittest.TestCase):
    def test_barrier_and_one_time_final_with_real_search(self):
        candidates, policies = {}, {}
        with PublicSupervisor(challenge_cap=160) as s:
            for arm in DEFAULT_ARMS:
                task=s.open_arm(arm)
                mode="adaptive" if "adaptive" in arm else arm
                policy=AdaptivePolicy() if mode=="adaptive" else None
                result=search(task,100,42,mode,policy=policy,train=False)
                candidates[arm]=result.selected.candidate
                policies[arm]=policy.snapshot() if policy else None
                receipt=s.freeze_arm(arm,candidates[arm],policies[arm])
                self.assertTrue(task.closed)
                self.assertEqual(receipt["learner"]["challenges"],task.challenge_requests)
                self.assertEqual(receipt["final_queries"],0)
                with self.assertRaises(EvaluatorUnavailable):
                    task.falsify(candidates[arm])
                if arm!=DEFAULT_ARMS[-1]:
                    with self.assertRaisesRegex(ValueError,"not all arms frozen"):
                        s.score_all(candidates,policies)
            result=s.score_all(candidates,policies)
            self.assertEqual(set(result["final_receipts"]),set(DEFAULT_ARMS))
            self.assertFalse(result["milestone_claim_eligible"])
            for arm, row in result["final_receipts"].items():
                self.assertEqual(row["final_queries"],1)
                self.assertEqual(row["barrier_sha256"],result["barrier_sha256"])
                self.assertEqual(row["exact_success"],1)
                self.assertEqual(row["mae"],0)
                self.assertEqual(candidates[arm].evaluate(9),19)
            with self.assertRaisesRegex(ValueError,"terminal"):
                s.score_all(candidates,policies)
            print("USF_PUBLIC_SUPERVISOR "+json.dumps({
                "arms":list(DEFAULT_ARMS),
                "challenge_requests":{a:r["learner"]["challenges"] for a,r in result["freeze_receipts"].items()},
                "exact_success":{a:r["exact_success"] for a,r in result["final_receipts"].items()},
                "final_queries":sum(r["final_queries"] for r in result["final_receipts"].values()),
                "public_fixture_only":True},sort_keys=True),flush=True)

    def test_changed_policy_or_candidate_never_reaches_final(self):
        for change in ("policy","candidate"):
            with self.subTest(change=change),PublicSupervisor(("one",)) as s:
                task=s.open_arm("one")
                task.training_samples()
                p={"count":0}
                h=known()
                s.freeze_arm("one",h,p)
                if change=="policy":
                    p["count"]=1
                else:
                    h=Expr.var()
                with patch.object(s,"_exchange",wraps=s._exchange) as channel:
                    with self.assertRaises(EvaluatorUnavailable):
                        s.score_all({"one":h},{"one":p})
                    channel.assert_not_called()
                self.assertTrue(s._closed)

    def test_worker_rejects_different_ast_even_if_supervisor_bypassed(self):
        with PublicSupervisor(("one",)) as s:
            task=s.open_arm("one")
            task.training_samples()
            frozen=s.freeze_arm("one",known(),None)
            with self.assertRaises((EOFError,OSError)):
                s._exchange("one",{"op":"VERIFY_FINAL","ast":["x"],
                                  "barrier":{"one":frozen["freeze_sha256"]}})
            self.assertEqual(task._proc.wait(timeout=2),2)

    def test_learner_forbidden_control_and_exhaustion_cannot_freeze(self):
        for mode in ("forbidden","cap"):
            with self.subTest(mode=mode),PublicSupervisor(("one",),challenge_cap=1) as s:
                task=s.open_arm("one")
                task.training_samples()
                if mode=="forbidden":
                    with self.assertRaises(EvaluatorUnavailable):
                        task._request("VERIFY_FINAL",{"ast":known().serial()})
                else:
                    self.assertIsNone(task.falsify(known()))
                    with self.assertRaises(EvaluatorUnavailable):
                        task.falsify(known())
                with self.assertRaises(EvaluatorUnavailable):
                    s.freeze_arm("one",known(),None)
                self.assertTrue(s._closed)
                self.assertIsNotNone(task._proc.poll())

    def test_wrong_candidate_can_be_scored_without_implicit_success(self):
        with PublicSupervisor(("one",)) as s:
            t=s.open_arm("one")
            t.training_samples()
            h=Expr.var()
            proof=s.freeze_arm("one",h,None)
            self.assertEqual(proof["learner"]["frames"],1)
            self.assertEqual(proof["learner"]["challenges"],0)
            r=s.score_all({"one":h},{"one":None})["final_receipts"]["one"]
            self.assertEqual(r["exact_success"],0)
            self.assertGreater(r["mae"],0)

    def test_roster_and_reuse_checks(self):
        for arms in ((),("one","one"),("one",True),("one! ",)):
            with self.subTest(arms=arms),self.assertRaises(ValueError):
                PublicSupervisor(arms)
        with PublicSupervisor(("one",)) as s:
            with self.assertRaises(ValueError):
                s.open_arm("other")
            s.open_arm("one")
            with self.assertRaises(ValueError):
                s.open_arm("one")


if __name__=="__main__":
    unittest.main()
