"""Adversarial regression controls for public supervisor protocol."""
import json
import socket
import time
import unittest
from unittest.mock import patch

from usf06.core import Expr, expr, sha256
from usf06.evaluator_client import EvaluatorUnavailable
from usf06.evaluator_supervisor import CONTROL, PublicSupervisor, send
from usf06.evaluator_worker import MAX_FRAME


def known():
    return expr("add",expr("mul",Expr.const(2),Expr.var()),Expr.const(1))


class ControlBoundaryTests(unittest.TestCase):
    def test_noncanonical_policy_snapshots_rejected(self):
        # The existing generic sha256 accepts nonfinite floats, coerces
        # integer object keys to strings, and treats tuples as arrays.
        for snapshot in ({"gain":float("nan")},{"gain":float("inf")},
                         {1:"value"}, {"counts":(1,2)}, {"gain":10**400}):
            with self.subTest(snapshot=str(snapshot)[:80]), PublicSupervisor(("one",)) as s:
                task=s.open_arm("one")
                task.training_samples()
                with self.assertRaises(EvaluatorUnavailable):
                    s.freeze_arm("one",known(),snapshot)
                self.assertTrue(s._closed)
                self.assertIsNotNone(task._proc.poll())

    def test_malformed_control_frames_fail_closed_in_actual_child(self):
        for variant in ("bad_json","duplicate","oversize","wrong_session",
                        "unknown_op","extra_field","wrong_version","bad_digest"):
            with self.subTest(variant=variant), PublicSupervisor(("one",)) as s:
                task=s.open_arm("one")
                task.training_samples()
                task._seal_transport()
                _,control=s._sessions["one"]
                control.settimeout(2)
                message={"version":CONTROL,"op":"FREEZE","session":task._session,
                         "candidate_sha256":sha256(known().serial()),
                         "policy_sha256":sha256(None),
                         "protocol_sha256":s.protocol_sha256}
                if variant=="bad_json":
                    raw=b"{\n"
                elif variant=="duplicate":
                    raw=(json.dumps(message)[:-1]+',"op":"FREEZE"}\n').encode()
                elif variant=="oversize":
                    raw=b" "*MAX_FRAME+b"\n"
                else:
                    if variant=="wrong_session": message["session"]="0"*32
                    elif variant=="unknown_op": message["op"]="REVEAL"
                    elif variant=="extra_field": message["target"]=True
                    elif variant=="wrong_version": message["version"]="wrong"
                    else: message["policy_sha256"]="broken"
                    raw=(json.dumps(message)+"\n").encode()
                try:
                    control.sendall(raw)
                    self.assertEqual(control.recv(4096),b"")
                except (BrokenPipeError,ConnectionResetError):
                    pass
                self.assertEqual(task._proc.wait(timeout=3),2)

    def test_forged_freeze_receipts_rejected_even_with_recomputed_hash(self):
        for variant in ("counter","session","candidate","extra","bool"):
            with self.subTest(variant=variant), PublicSupervisor(("one",)) as s:
                task=s.open_arm("one")
                task.training_samples()
                real=s._exchange
                def forged(arm,message):
                    r=real(arm,message)
                    f=r["receipt"]
                    if variant=="counter": f["learner"]["frames"]+=1
                    elif variant=="session": f["session"]="0"*32
                    elif variant=="candidate": f["candidate_sha256"]="a"*64
                    elif variant=="extra": f["target"]=["x"]
                    else: f["learner"]["observations"]=True
                    f["freeze_sha256"]=sha256({k:v for k,v in f.items() if k!="freeze_sha256"})
                    return r
                with patch.object(s,"_exchange",side_effect=forged):
                    with self.assertRaises(EvaluatorUnavailable):
                        s.freeze_arm("one",known(),None)
                self.assertTrue(s._closed)
                self.assertIsNotNone(task._proc.poll())

    def test_partial_arm_failure_releases_no_results(self):
        with PublicSupervisor(("one","two")) as s:
            for a in s.arms:
                t=s.open_arm(a)
                t.training_samples()
                s.freeze_arm(a,known(),None)
            failed=s._sessions["two"][0]._proc
            failed.kill()
            failed.wait(timeout=2)
            with self.assertRaises(EvaluatorUnavailable):
                s.score_all({"one":known(),"two":known()},{"one":None,"two":None})
            self.assertFalse(s._released)
            self.assertTrue(s._closed)
            self.assertTrue(all(t._proc.poll() is not None for t,_ in s._sessions.values()))

    def test_stalled_control_worker_times_out_and_is_reaped(self):
        # Stop an actual child after learner EOF. No control response exists.
        import os,signal
        with PublicSupervisor(("one",),request_timeout=0.5) as s:
            task=s.open_arm("one")
            task.training_samples()
            os.kill(task._proc.pid,signal.SIGSTOP)
            started=time.monotonic()
            with self.assertRaises(EvaluatorUnavailable):
                s.freeze_arm("one",known(),None)
            self.assertLess(time.monotonic()-started,5)
            self.assertTrue(s._closed)
            self.assertIsNotNone(task._proc.poll())

    def test_forged_final_receipts_fail_closed(self):
        for field,value in (("exact_success",True),("exact_fraction",0.5),
                            ("mae",-1),("final_queries",2),("barrier_sha256","a"*64)):
            with self.subTest(field=field), PublicSupervisor(("one",)) as s:
                t=s.open_arm("one")
                t.training_samples()
                s.freeze_arm("one",known(),None)
                real=s._exchange
                def forged(arm,message):
                    r=real(arm,message)
                    r[field]=value
                    return r
                with patch.object(s,"_exchange",side_effect=forged):
                    with self.assertRaises(EvaluatorUnavailable):
                        s.score_all({"one":known()},{"one":None})
                self.assertFalse(s._released)
                self.assertTrue(s._closed)


if __name__=="__main__":
    unittest.main()
