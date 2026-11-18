"""USF-EVAL-1 public synthetic evaluator worker (learner channel only).

Launch from a trusted supervisor as a separate process. Fixed public fixture,
not a secret task or a hostile-process sandbox. Final-release control,
supervisor resource receipts and transport timeouts are separate work.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from .core import Expr, SealedTask, expr

VERSION = "USF-EVAL-1"
MAX_FRAME = 65_536
MAX_NODES = 23
MAX_DEPTH = 9
MAX_CHALLENGES = 1600
REQUEST_KEYS = {"version", "session", "seq", "op", "payload"}
ARITY = {"x": 0, "neg": 1, "add": 2, "sub": 2, "mul": 2}


class ProtocolError(ValueError):
    pass


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate key")
        result[key] = value
    return result


def _constant(_):
    raise ProtocolError("nonfinite JSON")


def _decode(raw):
    if not raw.endswith(b"\n") or len(raw) > MAX_FRAME:
        raise ProtocolError("invalid frame")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("invalid JSON") from exc
    if type(value) is not dict:
        raise ProtocolError("expected object")
    return value


def parse_candidate(raw):
    """Validate arity/types and incrementally bounded AST before construction."""
    remaining = MAX_NODES

    def walk(node, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_DEPTH or type(node) is not list or not node:
            raise ProtocolError("invalid AST")
        op = node[0]
        if type(op) is not str:
            raise ProtocolError("invalid AST opcode")
        if op == "const":
            if len(node) != 2 or type(node[1]) is not int or abs(node[1]) > 1000:
                raise ProtocolError("invalid constant")
            return Expr.const(node[1])
        if op not in ARITY or len(node) != ARITY[op] + 1:
            raise ProtocolError("invalid AST arity")
        return Expr(op, tuple(walk(child, depth + 1) for child in node[1:]))

    return walk(raw, 1)


def _response(session, seq, status, payload):
    return {"version": VERSION, "session": session, "seq": seq,
            "status": status, "payload": payload}


def _write(out, message):
    raw = (json.dumps(message, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_FRAME:
        raise ProtocolError("response too large")
    out.write(raw)
    out.flush()


def serve(task, session, challenge_cap, inp, out):
    """One observe, then bounded challenge requests; any protocol error closes.

    The learner never receives detailed falsification data. Repeats consume
    the same request budget as unique candidates. This API has no final op.
    """
    if (type(session) is not str or re.fullmatch(r"[0-9a-f]{32}", session) is None
            or type(challenge_cap) is not int or not 1 <= challenge_cap <= MAX_CHALLENGES):
        return 2
    expected_seq = 1
    observed = False
    challenges = 0

    def fail(code):
        _write(out, _response(session, expected_seq, "ERROR", {"code": code}))
        return 2

    while True:
        raw = inp.readline(MAX_FRAME + 1)
        if not raw:
            return 0
        try:
            request = _decode(raw)
        except ProtocolError:
            return 2  # no trustworthy envelope: close silently
        if request.get("session") != session:
            return 2
        if set(request) != REQUEST_KEYS:
            return fail("BAD_FRAME")
        if request["version"] != VERSION:
            return fail("BAD_VERSION")
        seq = request["seq"]
        if type(seq) is not int or seq != expected_seq:
            return fail("BAD_SEQUENCE")
        if expected_seq > challenge_cap + 1:
            return fail("BUDGET_EXHAUSTED")
        op, payload = request["op"], request["payload"]
        if type(op) is not str or op not in ("OBSERVE", "CHALLENGE"):
            return fail("BAD_OP")
        if type(payload) is not dict:
            return fail("BAD_PAYLOAD")
        try:
            if op == "OBSERVE":
                if observed:
                    return fail("WRONG_PHASE")
                if payload:
                    return fail("BAD_PAYLOAD")
                pairs = task.training_samples()
                if not 1 <= len(pairs) <= 32 or any(
                    type(x) is not int or abs(x) > 10000 or
                    type(y) is not int or abs(y) > 10**12 for x, y in pairs
                ):
                    return fail("EVALUATOR_FAILURE")
                response = {"samples": [list(pair) for pair in pairs]}
                observed = True
            else:
                if not observed:
                    return fail("WRONG_PHASE")
                if challenges >= challenge_cap:
                    return fail("BUDGET_EXHAUSTED")
                challenges += 1
                if set(payload) != {"ast"}:
                    return fail("BAD_PAYLOAD")
                try:
                    candidate = parse_candidate(payload["ast"])
                except ProtocolError:
                    return fail("INVALID_AST")
                detail = task.falsify(candidate)
                verdict = ("PASS" if detail is None else
                           "INVALID" if detail.get("invalid") is True else "FAIL")
                response = {"verdict": verdict}
        except Exception:
            # No raw exception or evaluator values on the learner channel.
            return fail("EVALUATOR_FAILURE")
        _write(out, _response(session, seq, "OK", response))
        expected_seq += 1


def _public_fixture():
    x = Expr.var()
    target = expr("add", expr("mul", Expr.const(2), x), Expr.const(1))
    return SealedTask("public-evaluator-fixture", "linear", target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--challenge-cap", type=int, default=MAX_CHALLENGES)
    args = parser.parse_args(argv)
    try:
        return serve(_public_fixture(), args.session, args.challenge_cap,
                     sys.stdin.buffer, sys.stdout.buffer)
    except (OSError, ProtocolError, ValueError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
