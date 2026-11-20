"""Target-free search adapter for the public USF-EVAL-1 evaluator worker.

POSIX pipe transport; fail closed on missing/invalid replies or deadlines.
No protected task administration, supervisor final channel or OS sandbox.
The public fixture is deliberately reproducible and its code is not secret.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import select
import subprocess
import sys
import time
from typing import Protocol

from .core import Expr
from .evaluator_worker import VERSION, MAX_FRAME, MAX_CHALLENGES, _decode


class ObservationTask(Protocol):
    task_id: str

    def training_samples(self) -> tuple[tuple[int, int], ...]: ...
    def falsify(self, candidate: Expr) -> dict | None: ...


class EvaluatorUnavailable(RuntimeError):
    """Terminal transport/protocol failure; never a successful verdict."""


def _spawn(session, cap):
    return subprocess.Popen(
        [sys.executable, "-m", "usf06.evaluator_worker", "--session", session,
         "--challenge-cap", str(cap)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=0, close_fds=True,
    )


class PublicEvaluatorTask:
    """Single-session observation/verdict adapter; do not reuse across arms."""

    def __init__(self, *, challenge_cap=1600, request_timeout=2.0,
                 session_timeout=60.0, _spawn_worker=None):
        if os.name != "posix":
            raise NotImplementedError("POSIX pipe transport required")
        if type(challenge_cap) is not int or not 1 <= challenge_cap <= MAX_CHALLENGES:
            raise ValueError("invalid challenge cap")
        for value in (request_timeout, session_timeout):
            if type(value) not in (float, int) or not 0 < value <= 120:
                raise ValueError("invalid timeout")
        self.task_id = "public-evaluator-fixture"
        self._session = secrets.token_hex(16)
        self._cap = challenge_cap
        self._timeout = float(request_timeout)
        self._expires = time.monotonic() + session_timeout
        self._seq = 0
        self._samples = None
        self._closed = False
        self._challenges = 0
        self._request_bytes = 0
        self._proc = (_spawn if _spawn_worker is None else _spawn_worker)(self._session, challenge_cap)
        try:
            os.set_blocking(self._proc.stdin.fileno(), False)
            os.set_blocking(self._proc.stdout.fileno(), False)
        except BaseException:
            self.close()
            raise

    @property
    def challenge_requests(self):
        return self._challenges

    @property
    def request_bytes(self):
        """Bytes actually written to the learner pipe, including partial writes."""
        return self._request_bytes

    @property
    def closed(self):
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for pipe in (self._proc.stdin, self._proc.stdout):
            if pipe is not None:
                pipe.close()
        try:
            self._proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def _seal_transport(self):
        """Trusted supervisor only: irreversibly close the learner pipes."""
        if self._closed or self._samples is None or time.monotonic() >= self._expires:
            self._abort()
        self._closed = True
        self._proc.stdin.close()
        self._proc.stdout.close()

    def _abort(self):
        self.close()
        raise EvaluatorUnavailable("evaluator session failed")

    def _remaining(self, end):
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    def _request(self, op, payload):
        if self._closed:
            raise EvaluatorUnavailable("evaluator session closed")
        end = min(self._expires, time.monotonic() + self._timeout)
        self._seq += 1
        try:
            message = {"version": VERSION, "session": self._session,
                       "seq": self._seq, "op": op, "payload": payload}
            raw = (json.dumps(message, separators=(",", ":"), allow_nan=False)
                   + "\n").encode("utf-8")
            if len(raw) > MAX_FRAME:
                raise ValueError
            fd = self._proc.stdin.fileno()
            sent = 0
            while sent < len(raw):
                if not select.select([], [fd], [], self._remaining(end))[1]:
                    raise TimeoutError
                try:
                    written = os.write(fd, raw[sent:])
                    if written <= 0:
                        raise OSError("zero-byte pipe write")
                    sent += written
                    self._request_bytes += written
                except BlockingIOError:
                    continue
            received = bytearray()
            fd = self._proc.stdout.fileno()
            while b"\n" not in received:
                if not select.select([fd], [], [], self._remaining(end))[0]:
                    raise TimeoutError
                try:
                    part = os.read(fd, min(4096, MAX_FRAME + 1 - len(received)))
                except BlockingIOError:
                    continue
                if not part:
                    raise EOFError
                received.extend(part)
                if len(received) > MAX_FRAME:
                    raise ValueError
            if received.count(b"\n") != 1 or not received.endswith(b"\n"):
                raise ValueError
            r = _decode(bytes(received))
            if (set(r) != {"version", "session", "seq", "status", "payload"}
                    or r["version"] != VERSION or r["session"] != self._session
                    or type(r["seq"]) is not int or r["seq"] != self._seq
                    or r["status"] != "OK" or type(r["payload"]) is not dict):
                raise ValueError
            return r["payload"]
        except (OSError, ValueError, EOFError, TypeError, RecursionError, OverflowError):
            self._abort()

    def training_samples(self):
        if self._closed:
            raise EvaluatorUnavailable("evaluator session closed")
        if self._samples is None:
            r = self._request("OBSERVE", {})
            samples = r.get("samples")
            if (set(r) != {"samples"} or type(samples) is not list
                    or not 1 <= len(samples) <= 32
                    or any(type(p) is not list or len(p) != 2
                           or type(p[0]) is not int or abs(p[0]) > 10000
                           or type(p[1]) is not int or abs(p[1]) > 10**12
                           for p in samples)):
                self._abort()
            self._samples = tuple(tuple(p) for p in samples)
        return self._samples

    def falsify(self, candidate):
        if self._closed:
            raise EvaluatorUnavailable("evaluator session closed")
        if self._samples is None or self._challenges >= self._cap:
            self._abort()
        self._challenges += 1
        r = self._request("CHALLENGE", {"ast": candidate.serial()})
        verdict = r.get("verdict")
        if set(r) != {"verdict"} or verdict not in ("PASS", "FAIL", "INVALID"):
            self._abort()
        # The legacy search consumes only None versus non-None for selection.
        # Do not fabricate challenge coordinates, residuals or expected values.
        return None if verdict == "PASS" else {"verdict": verdict}
