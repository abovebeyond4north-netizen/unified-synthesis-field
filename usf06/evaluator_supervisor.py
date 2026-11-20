"""Public synthetic supervisor barrier for USF-EVAL-1.

The supervisor owns a separate inherited socket; the learner adapter receives
only its subprocess pipes. This is a trusted local API, not hostile-code/OS
isolation. The fixture and source are public. All counters are request-level
receipts, not measured compute or independent attestations.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import socket
import subprocess
import sys
import time

from .core import sha256
from .evaluator_client import PublicEvaluatorTask, EvaluatorUnavailable
from .evaluator_worker import (
    VERSION, MAX_FRAME, MAX_CHALLENGES, _decode, parse_candidate,
    _public_fixture, serve,
)

CONTROL = "USF-SUPERVISOR-1"
DEFAULT_ARMS = ("random", "evolution", "cold_adaptive", "adaptive")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def is_digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def send(sock, obj, deadline):
    raw = (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode()
    require(len(raw) <= MAX_FRAME, "control frame too large")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    sock.settimeout(remaining)
    sock.sendall(raw)


def receive(sock, deadline):
    raw = bytearray()
    while b"\n" not in raw:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        sock.settimeout(remaining)
        part = sock.recv(min(4096, MAX_FRAME + 1 - len(raw)))
        if not part:
            raise EOFError
        raw.extend(part)
        require(len(raw) <= MAX_FRAME, "control frame too large")
    require(raw.count(b"\n") == 1 and raw.endswith(b"\n"), "invalid control frame")
    return _decode(bytes(raw))


def policy_digest(snapshot):
    """Hash only bounded, strict JSON policy state, without type coercion.

    Dict keys must be strings; tuple/list and integer/string key aliases are
    not admitted. Bounds limit accidental excessive serialization, not an OS
    security boundary. Use the same encoding at freeze and final submission.
    """
    nodes = 0
    def validate(value, depth):
        nonlocal nodes
        nodes += 1
        require(nodes <= 2048 and depth <= 16, "policy structure bounds")
        t = type(value)
        if value is None or t is bool:
            return
        if t is int:
            require(-(2**63) <= value <= 2**63-1, "policy integer bounds")
        elif t is float:
            require(math.isfinite(value), "nonfinite policy number")
        elif t is str:
            require(len(value) <= 4096, "policy string bounds")
        elif t is list:
            require(len(value) <= 2048, "policy array bounds")
            for item in value:
                validate(item, depth+1)
        elif t is dict:
            require(len(value) <= 2048, "policy object bounds")
            for key, item in value.items():
                require(type(key) is str and len(key) <= 4096, "policy key")
                validate(item, depth+1)
        else:
            raise ValueError("non-JSON policy type")
    validate(snapshot, 1)
    raw=json.dumps(snapshot,sort_keys=True,separators=(",",":"),ensure_ascii=True,
                   allow_nan=False).encode("utf-8")
    require(len(raw) <= MAX_FRAME, "policy byte bounds")
    # Deliberately matches core.sha256 on the admitted JSON domain.
    return sha256(snapshot)


def candidate_ast(candidate):
    ast = candidate.serial()
    return parse_candidate(ast).serial()


def worker(session, cap, fd, lifetime):
    # The learner channel must reach clean EOF before any control request is
    # consumed. The target stays in this child for both phases.
    start = time.monotonic()
    expires = start + lifetime
    receipt = {}
    task = _public_fixture()
    with socket.socket(fileno=fd) as control:
        try:
            code = serve(task, session, cap, sys.stdin.buffer, sys.stdout.buffer,
                         receipt=receipt)
            if code != 0 or receipt.get("observations") != 1:
                return 2
            freeze = receive(control, expires)
            require(set(freeze) == {"version","op","session","candidate_sha256",
                                    "policy_sha256","protocol_sha256"}, "bad freeze")
            require(freeze["version"] == CONTROL and freeze["op"] == "FREEZE"
                    and freeze["session"] == session, "bad freeze")
            require(all(is_digest(freeze[k]) for k in ("candidate_sha256","policy_sha256",
                                                      "protocol_sha256")), "bad commitment")
            body = {
                "session": session, "candidate_sha256": freeze["candidate_sha256"],
                "policy_sha256": freeze["policy_sha256"],
                "protocol_sha256": freeze["protocol_sha256"],
                "learner": receipt, "challenge_cap": cap, "final_queries": 0,
            }
            frozen = {**body, "freeze_sha256": sha256(body)}
            send(control, {"status":"FROZEN","receipt":frozen}, expires)
            request = receive(control, expires)
            require(set(request) == {"version","op","session","ast","barrier"},
                    "bad final request")
            require(request["version"] == CONTROL and request["op"] == "VERIFY_FINAL"
                    and request["session"] == session, "bad final request")
            barrier = request["barrier"]
            require(type(barrier) is dict and 1 <= len(barrier) <= 16 and
                    all(type(k) is str and is_digest(v) for k,v in barrier.items())
                    and len(set(barrier.values())) == len(barrier)
                    and frozen["freeze_sha256"] in barrier.values(), "bad barrier")
            ast = parse_candidate(request["ast"])
            require(sha256(ast.serial()) == freeze["candidate_sha256"], "candidate changed")
            result = task.holdout_score(ast)
            # Only the supervisor socket receives this response; exit after one.
            send(control, {
                "status":"FINAL", "freeze_sha256":frozen["freeze_sha256"],
                "barrier_sha256":sha256(barrier), "final_queries":1,
                "exact_success":int(result.exact_fraction == 1.0),
                "exact_fraction":result.exact_fraction, "mae":result.mae,
            }, expires)
            return 0
        except (OSError, ValueError, TypeError, EOFError, KeyError, RecursionError):
            return 2


class PublicSupervisor:
    """Trusted supervisor for one public task and a declared set of arms."""

    def __init__(self, arms=DEFAULT_ARMS, *, challenge_cap=1600,
                 request_timeout=2.0, session_timeout=60.0):
        require(type(arms) in (tuple,list) and 1 <= len(arms) <= 16 and
                all(type(a) is str and re.fullmatch("[a-z_]{1,32}",a) for a in arms)
                and len(set(arms)) == len(arms), "invalid arm roster")
        require(type(challenge_cap) is int and 1 <= challenge_cap <= MAX_CHALLENGES,
                "invalid challenge cap")
        for v in (request_timeout,session_timeout):
            require(type(v) in (int,float) and 0 < v <= 120, "invalid timeout")
        self.arms = tuple(arms)
        self.cap = challenge_cap
        self.timeout = request_timeout
        self.lifetime = session_timeout
        self.protocol_sha256 = sha256({
            "version":VERSION, "control":CONTROL, "arms":self.arms,
            "fixture":"public-evaluator-fixture", "challenge_cap":challenge_cap,
            "request_timeout":request_timeout,"session_timeout":session_timeout,
            "max_frame":MAX_FRAME,
        })
        self._sessions = {}
        self._frozen = {}
        self._closed = False
        self._released = False

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for task, control in self._sessions.values():
            control.close()
            task.close()
            p = task._proc
            try:
                p.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def _active(self):
        require(not self._closed and not self._released, "supervisor terminal")

    def open_arm(self, arm):
        self._active()
        require(arm in self.arms and arm not in self._sessions, "invalid or reused arm")
        parent, child = socket.socketpair()
        def spawn(session,cap):
            return subprocess.Popen(
                [sys.executable,"-m","usf06.evaluator_supervisor",
                 "--session",session,"--cap",str(cap),"--fd",str(child.fileno()),
                 "--lifetime",str(self.lifetime)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                bufsize=0,close_fds=True,pass_fds=(child.fileno(),))
        try:
            task = PublicEvaluatorTask(challenge_cap=self.cap,request_timeout=self.timeout,
                                       session_timeout=self.lifetime,_spawn_worker=spawn)
        except BaseException:
            parent.close()
            self.close()
            raise
        finally:
            child.close()
        self._sessions[arm] = (task,parent)
        return task

    def _exchange(self, arm, message):
        task, control = self._sessions[arm]
        deadline = min(task._expires,time.monotonic()+self.timeout)
        send(control,{"version":CONTROL,"session":task._session,**message},deadline)
        return receive(control,deadline)

    def freeze_arm(self, arm, candidate, policy_snapshot):
        self._active()
        require(arm in self._sessions and arm not in self._frozen,"invalid freeze arm")
        try:
            task,_ = self._sessions[arm]
            ast = candidate_ast(candidate)
            ch,ph = sha256(ast),policy_digest(policy_snapshot)
            task._seal_transport()
            r = self._exchange(arm, {"op":"FREEZE","candidate_sha256":ch,
                "policy_sha256":ph,"protocol_sha256":self.protocol_sha256})
            require(set(r)=={"status","receipt"} and r["status"]=="FROZEN","freeze missing")
            f=r["receipt"]
            require(type(f) is dict and set(f)=={"session","candidate_sha256","policy_sha256",
                    "protocol_sha256","learner","challenge_cap","final_queries","freeze_sha256"},
                    "invalid freeze receipt")
            body={k:v for k,v in f.items() if k!="freeze_sha256"}
            require(f["freeze_sha256"]==sha256(body) and f["session"]==task._session
                    and f["candidate_sha256"]==ch and f["policy_sha256"]==ph
                    and f["protocol_sha256"]==self.protocol_sha256
                    and type(f["challenge_cap"]) is int and f["challenge_cap"]==self.cap
                    and type(f["final_queries"]) is int and f["final_queries"]==0,
                    "freeze receipt mismatch")
            counters=f["learner"]
            require(type(counters) is dict and set(counters)=={
                    "frames","request_bytes","observations","challenges"} and
                    all(type(v) is int and v>=0 for v in counters.values()) and
                    counters["observations"]==1 and
                    counters["frames"]==1+counters["challenges"] and
                    counters["challenges"]==task.challenge_requests<=self.cap and
                    0<counters["request_bytes"]==task.request_bytes<=counters["frames"]*MAX_FRAME,
                    "learner receipt mismatch")
            self._frozen[arm]=f
            return json.loads(json.dumps(f))
        except (OSError,ValueError,EOFError,TypeError,KeyError,EvaluatorUnavailable):
            self.close()
            raise EvaluatorUnavailable("supervisor freeze failed") from None

    def score_all(self, candidates, policies):
        self._active()
        # No score request until every arm freezes. A premature caller may
        # finish the remaining arms, but no feedback has been released.
        require(set(self._frozen)==set(self.arms),"not all arms frozen")
        try:
            require(set(candidates)==set(policies)==set(self.arms),"incomplete final roster")
            asts={a:candidate_ast(candidates[a]) for a in self.arms}
            for a in self.arms:
                require(sha256(asts[a])==self._frozen[a]["candidate_sha256"] and
                        policy_digest(policies[a])==self._frozen[a]["policy_sha256"],
                        "post-freeze candidate or policy changed")
            barrier={a:self._frozen[a]["freeze_sha256"] for a in self.arms}
            results={}
            for a in self.arms:
                r=self._exchange(a,{"op":"VERIFY_FINAL","ast":asts[a],"barrier":barrier})
                require(set(r)=={"status","freeze_sha256","barrier_sha256","final_queries",
                                "exact_success","exact_fraction","mae"} and r["status"]=="FINAL"
                        and r["freeze_sha256"]==barrier[a] and r["barrier_sha256"]==sha256(barrier)
                        and type(r["final_queries"]) is int and r["final_queries"]==1
                        and type(r["exact_success"]) is int and r["exact_success"] in (0,1)
                        and type(r["exact_fraction"]) in (int,float) and 0<=r["exact_fraction"]<=1
                        and r["exact_success"]==int(r["exact_fraction"]==1)
                        and type(r["mae"]) in (int,float) and 0<=r["mae"]<=2*10**12
                        and (r["mae"]==0)==(r["exact_fraction"]==1),
                        "invalid final receipt")
                task,_=self._sessions[a]
                require(task._proc.wait(timeout=self.timeout)==0,"worker failed after final")
                results[a]=r
            self._released=True
            return {"protocol_sha256":self.protocol_sha256,"barrier_sha256":sha256(barrier),
                    "freeze_receipts":json.loads(json.dumps(self._frozen)),
                    "final_receipts":results,"independent_provenance_authenticated":False,
                    "matched_actual_compute_established":False,"milestone_claim_eligible":False}
        except (OSError,ValueError,EOFError,TypeError,KeyError,subprocess.TimeoutExpired):
            self.close()
            raise EvaluatorUnavailable("supervisor final evaluation failed") from None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session",required=True)
    p.add_argument("--cap",type=int,required=True)
    p.add_argument("--fd",type=int,required=True)
    p.add_argument("--lifetime",type=float,required=True)
    a=p.parse_args()
    return worker(a.session,a.cap,a.fd,a.lifetime)


if __name__=="__main__":
    raise SystemExit(main())
