"""Real child-process tests, synthetic fixture only."""
import json
import subprocess
import sys
import unittest

from usf06.evaluator_worker import VERSION, MAX_FRAME

SESSION="ab"*16
TARGET=["add",["mul",["const",2],["x"]],["const",1]]


def request(seq, op, payload=None, **changes):
    return {"version":VERSION,"session":SESSION,"seq":seq,"op":op,
            "payload":{} if payload is None else payload,**changes}


def child(messages, cap=3):
    frames=b"".join((json.dumps(m)+"\n").encode() if isinstance(m,dict) else m
                    for m in messages)
    p=subprocess.run([sys.executable,"-m","usf06.evaluator_worker",
                      "--session",SESSION,"--challenge-cap",str(cap)],
                     input=frames,capture_output=True,timeout=5,check=False)
    return p, [json.loads(line) for line in p.stdout.splitlines()]


class EvaluatorWorkerTests(unittest.TestCase):
    def check_error(self, messages, code, cap=3):
        p,rows=child(messages,cap)
        self.assertEqual(p.returncode,2)
        self.assertEqual(p.stderr,b"")
        self.assertEqual(rows[-1]["status"],"ERROR")
        self.assertEqual(rows[-1]["payload"],{"code":code})
        return rows

    def test_process_observations_and_redacted_verdicts(self):
        overflow=["mul",["mul",["const",1000],["const",1000]],
                  ["mul",["mul",["const",1000],["const",1000]],["x"]]]
        p,rows=child([request(1,"OBSERVE"),request(2,"CHALLENGE",{"ast":TARGET}),
                      request(3,"CHALLENGE",{"ast":["x"]}),
                      request(4,"CHALLENGE",{"ast":overflow})])
        self.assertEqual(p.returncode,0)
        self.assertEqual(p.stderr,b"")
        self.assertEqual([r["payload"] for r in rows],[
            {"samples":[[-2,-3],[-1,-1],[0,1],[1,3],[2,5]]},
            {"verdict":"PASS"},{"verdict":"FAIL"},{"verdict":"INVALID"}])
        for i,row in enumerate(rows,1):
            self.assertEqual(set(row),{"version","session","seq","status","payload"})
            self.assertEqual((row["version"],row["session"],row["seq"],row["status"]),
                             (VERSION,SESSION,i,"OK"))

    def test_replay_consumes_budget_and_closes(self):
        rows=self.check_error([request(1,"OBSERVE"),
            *[request(i,"CHALLENGE",{"ast":TARGET}) for i in range(2,5)],
            request(5,"CHALLENGE",{"ast":TARGET})],"BUDGET_EXHAUSTED")
        self.assertEqual(len(rows),5)
        self.assertTrue(all(r["payload"]=={"verdict":"PASS"} for r in rows[1:4]))

    def test_forbidden_ops_and_wrong_phase(self):
        for op in ("REVEAL","VERIFY_FINAL","FREEZE","READ_FILE","RESET"):
            with self.subTest(op=op):
                self.check_error([request(1,"OBSERVE"),request(2,op)],"BAD_OP")
        self.check_error([request(1,"CHALLENGE",{"ast":TARGET})],"WRONG_PHASE")
        self.check_error([request(1,"OBSERVE"),request(2,"OBSERVE")],"WRONG_PHASE")
        self.check_error([request(1,"OBSERVE",{"target":True})],"BAD_PAYLOAD")
        self.check_error([request(1,"OBSERVE"),request(2,"CHALLENGE",
                          {"ast":TARGET,"path":"/tmp"})],"BAD_PAYLOAD")

    def test_malformed_frames_and_session_close_without_diagnostics(self):
        for raw in (b"\xff\n",b"{\n",b"[]\n",b" "+b"x"*MAX_FRAME+b"\n",
                    b'{"session":"a","session":"b"}\n',
                    b'{"seq":NaN}\n',b'{"seq":1}'):
            with self.subTest(raw=raw[:32]):
                p,rows=child([raw,request(1,"OBSERVE")])
                self.assertEqual((p.returncode,p.stderr,rows),(2,b"",[]))
        p,rows=child([request(1,"OBSERVE",session="00"*16)])
        self.assertEqual((p.returncode,p.stderr,rows),(2,b"",[]))

    def test_strict_structure_and_ast_limits(self):
        self.check_error([request(1,"OBSERVE",version="bad")],"BAD_VERSION")
        self.check_error([request(True,"OBSERVE")],"BAD_SEQUENCE")
        self.check_error([request(1,"OBSERVE",extra="x")],"BAD_FRAME")
        self.check_error([request(1,"OBSERVE"),request(1,"CHALLENGE",{"ast":TARGET})],
                         "BAD_SEQUENCE")
        deep=["x"]
        for _ in range(10):
            deep=["neg",deep]
        wide=["x"]
        for _ in range(4):
            wide=["add",wide,wide]
        for ast in (["const",True],["const",1001],["x",1],["exec","x"],
                    ["add",["x"]],deep,wide):
            with self.subTest(ast=str(ast)[:80]):
                self.check_error([request(1,"OBSERVE"),request(2,"CHALLENGE",{"ast":ast})],
                                 "INVALID_AST")


if __name__=="__main__":
    unittest.main()
