"""Real search through the public worker and terminal transport controls."""
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from usf06.core import Expr
from usf06.evaluator_client import PublicEvaluatorTask, EvaluatorUnavailable
from usf06.evaluator_worker import _public_fixture
from usf06.search import AdaptivePolicy, search


def replacement_child(script):
    def spawn(session, cap):
        return subprocess.Popen([sys.executable, "-c", script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0, close_fds=True)
    return spawn


class EvaluatorAdapterTests(unittest.TestCase):
    def test_real_search_equivalence_and_target_free_interface(self):
        for mode in ("random","evolution","adaptive"):
            with self.subTest(mode=mode), PublicEvaluatorTask(challenge_cap=160) as task:
                self.assertFalse(hasattr(task,"_target"))
                self.assertFalse(hasattr(task,"holdout_score"))
                self.assertFalse(hasattr(task,"reveal"))
                self.assertFalse(hasattr(task,"family"))
                kwargs={"policy":AdaptivePolicy()} if mode=="adaptive" else {}
                remote=search(task,100,42,mode,**kwargs)
                self.assertGreater(task.challenge_requests,0)
                self.assertLessEqual(task.challenge_requests,remote.proposals)
                self.assertEqual(task.training_samples(),((-2,-3),(-1,-1),(0,1),(1,3),(2,5)))
                local=search(_public_fixture(),100,42,mode,**kwargs)
                self.assertEqual(remote.selected.candidate,local.selected.candidate)
                self.assertEqual(remote.selected.survived,local.selected.survived)
                self.assertEqual((remote.proposals,remote.attempts,remote.survived_count),
                                 (local.proposals,local.attempts,local.survived_count))

    def test_redaction_cap_and_session_cleanup(self):
        with PublicEvaluatorTask(challenge_cap=2) as task:
            before=task.training_samples()
            self.assertEqual(before,task.training_samples())
            self.assertEqual(task.falsify(Expr.var()),{"verdict":"FAIL"})
            from usf06.core import expr
            target=expr("add",expr("mul",Expr.const(2),Expr.var()),Expr.const(1))
            self.assertIsNone(task.falsify(target))
            with self.assertRaises(EvaluatorUnavailable):
                task.falsify(target)
            self.assertEqual(task.challenge_requests,2)
            self.assertTrue(task.closed)
            self.assertIsNotNone(task._proc.poll())
            with self.assertRaises(EvaluatorUnavailable):
                task.training_samples()

    def test_timeout_eof_and_invalid_reply_fail_closed(self):
        cases=(
            "import time; time.sleep(30)",
            "raise SystemExit(0)",
            "import sys; sys.stdin.readline(); print('not json',flush=True)",
            "import sys,json; m=json.loads(sys.stdin.readline()); "
            "print(json.dumps(dict(version=m['version'],session=m['session'],seq=m['seq'],"
            "status='OK',payload={'samples':[[0,1]],'target':'forbidden'})),flush=True)",
        )
        for script in cases:
            with self.subTest(script=script[:40]), patch(
                "usf06.evaluator_client._spawn", side_effect=replacement_child(script)
            ), PublicEvaluatorTask(request_timeout=0.2) as task:
                with self.assertRaisesRegex(EvaluatorUnavailable,"session failed"):
                    task.training_samples()
                self.assertTrue(task.closed)
                self.assertIsNotNone(task._proc.poll())

    def test_session_lifetime_and_observe_phase_fail_closed(self):
        with PublicEvaluatorTask(session_timeout=0.001) as task:
            import time
            time.sleep(0.01)
            with self.assertRaises(EvaluatorUnavailable):
                task.training_samples()
            self.assertTrue(task.closed)
        with PublicEvaluatorTask() as task:
            with self.assertRaises(EvaluatorUnavailable):
                task.falsify(Expr.var())
            self.assertTrue(task.closed)

    def test_invalid_configuration_rejected(self):
        for kw in ({"challenge_cap":True},{"challenge_cap":1601},
                   {"request_timeout":float("nan")},{"session_timeout":0}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                PublicEvaluatorTask(**kw)


if __name__=="__main__":
    unittest.main()
