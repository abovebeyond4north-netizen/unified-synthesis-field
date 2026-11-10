"""Deterministic symbolic-expression substrate, two-part code lengths, and sealed tasks.

The numerical checks here are explicitly EMPIRICAL, never formal proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: int | None = None

    def __post_init__(self) -> None:
        expected = {"const": 0, "x": 0, "add": 2, "sub": 2, "mul": 2, "neg": 1}
        if self.op not in expected or len(self.args) != expected[self.op]:
            raise ValueError("Unknown opcode or invalid arity")
        if self.op == "const":
            if type(self.value) is not int or abs(self.value) > 1000:
                raise ValueError("Constants must be bounded integers")
        elif self.value is not None:
            raise ValueError("Only const expressions have a value")

    @staticmethod
    def const(value: int) -> "Expr":
        return Expr("const", value=value)

    @staticmethod
    def var() -> "Expr":
        return Expr("x")

    def serial(self):
        return [self.op, self.value] if self.op == "const" else [self.op, *[a.serial() for a in self.args]]

    @classmethod
    def from_serial(cls, data):
        if not isinstance(data, list) or not data:
            raise ValueError("Invalid AST encoding")
        if data[0] == "const":
            if len(data) != 2:
                raise ValueError("Invalid const encoding")
            return cls.const(data[1])
        if data[0] == "x":
            if len(data) != 1:
                raise ValueError("Invalid x encoding")
            return cls.var()
        return cls(data[0], tuple(cls.from_serial(v) for v in data[1:]))

    def key(self) -> str:
        return canonical_json(self.serial())

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def depth(self) -> int:
        return 1 + max((a.depth() for a in self.args), default=0)

    def evaluate(self, x: int) -> int:
        if type(x) is not int or abs(x) > 10000:
            raise ValueError("Invalid evaluation point")
        if self.size() > 100 or self.depth() > 30:
            raise ValueError("Expression too large")
        return self._eval(x)

    def _eval(self, x: int) -> int:
        if self.op == "const":
            out = self.value
        elif self.op == "x":
            out = x
        elif self.op == "neg":
            out = -self.args[0]._eval(x)
        else:
            a, b = (arg._eval(x) for arg in self.args)
            if self.op == "add":
                out = a + b
            elif self.op == "sub":
                out = a - b
            else:
                out = a * b
        if abs(out) > 10**12:
            raise OverflowError("Bounded arithmetic exceeded")
        return out

    def to_lean(self) -> str:
        if self.op == "const":
            return f"({self.value} : ℤ)"
        if self.op == "x":
            return "x"
        if self.op == "neg":
            return f"(-{self.args[0].to_lean()})"
        op = {"add": "+", "sub": "-", "mul": "*"}[self.op]
        return f"({self.args[0].to_lean()} {op} {self.args[1].to_lean()})"

    def __str__(self):
        if self.op == "const":
            return str(self.value)
        if self.op == "x":
            return "x"
        if self.op == "neg":
            return f"(-{self.args[0]})"
        return f"({self.args[0]} {self.op} {self.args[1]})"


def expr(op: str, *children: Expr) -> Expr:
    # Algebra-preserving deterministic canonicalization; never used as a proof.
    if op in {"add", "mul"} and children[0].key() > children[1].key():
        children = (children[1], children[0])
    a = children[0]
    b = children[1] if len(children) == 2 else None
    if op == "neg":
        if a.op == "const":
            return Expr.const(-a.value)
        if a.op == "neg":
            return a.args[0]
    if b is not None:
        if a.op == "const" and b.op == "const":
            n = {"add": lambda: a.value + b.value,
                 "sub": lambda: a.value - b.value,
                 "mul": lambda: a.value * b.value}[op]()
            if abs(n) <= 1000:
                return Expr.const(n)
        if op == "add":
            if a == Expr.const(0):
                return b
            if b == Expr.const(0):
                return a
        if op == "sub" and b == Expr.const(0):
            return a
        if op == "mul":
            if a == Expr.const(0) or b == Expr.const(0):
                return Expr.const(0)
            if a == Expr.const(1):
                return b
            if b == Expr.const(1):
                return a
    return Expr(op, tuple(children))


def universal_integer_bits(n: int) -> int:
    """Prefix-free Elias gamma code length for zig-zag signed integers.

    zigzag: 0=>0, -1=>1, 1=>2, -2=>3, 2=>4, ...;
    code the strictly positive number zigzag(n)+1.
    """
    z = (-2 * n - 1) if n < 0 else (2 * n)
    return 2 * (z + 1).bit_length() - 1


def hypothesis_bits(h: Expr) -> int:
    """Two-part model code: 3 bits per opcode (six opcodes), integer gamma payload."""
    return 3 + (universal_integer_bits(h.value) if h.op == "const" else 0) + sum(
        hypothesis_bits(a) for a in h.args
    )


@dataclass(frozen=True)
class Fit:
    exact_fraction: float
    mae: float
    mdl_bits: int
    compression_gain_bits: int


def fit(h: Expr, samples: Iterable[tuple[int, int]]) -> Fit:
    rows = list(samples)
    if not rows:
        raise ValueError("Cannot fit on empty sample set")
    residuals = []
    for x, y in rows:
        try:
            residuals.append(y - h.evaluate(x))
        except (ValueError, OverflowError):
            # Semantically invalid candidates are assigned a massive but finite cost.
            residuals.append(10**12)
    model_bits = hypothesis_bits(h)
    data_bits = sum(universal_integer_bits(r) for r in residuals)
    null_bits = sum(universal_integer_bits(y) for _, y in rows)
    return Fit(sum(r == 0 for r in residuals) / len(rows),
               sum(abs(r) for r in residuals) / len(rows),
               model_bits + data_bits,
               null_bits - (model_bits + data_bits))


@dataclass(frozen=True)
class SealedTask:
    """Target expression is evaluator-owned; search sees observations and verdicts only."""
    task_id: str
    family: str
    _target: Expr
    train_x: tuple[int, ...] = (-2, -1, 0, 1, 2)
    challenge_x: tuple[int, ...] = (-6, -4, 3, 5)
    holdout_x: tuple[int, ...] = (-9, -7, 7, 9)

    def training_samples(self) -> tuple[tuple[int, int], ...]:
        return tuple((x, self._target.evaluate(x)) for x in self.train_x)

    def falsify(self, candidate: Expr):
        """Independent candidate-specific edge probes; NEVER queries holdout points.

        The test generator maximizes observed absolute discrepancy over a bounded
        candidate-independent challenge grid plus seeded candidate-specific probes.
        This is a bounded *empirical* falsifier, not a completeness guarantee.
        """
        import random
        rng = random.Random(int(sha256([self.task_id, candidate.key()])[:16], 16))
        forbidden = set(self.train_x) | set(self.holdout_x)
        generated = [x for x in range(-12, 13) if x not in forbidden]
        rng.shuffle(generated)
        points = list(dict.fromkeys([x for x in self.challenge_x if x not in forbidden]
                                    + generated[:8]))
        worst = None
        worst_error = 0
        for x in points:
            want = self._target.evaluate(x)
            try:
                got = candidate.evaluate(x)
            except (OverflowError, ValueError):
                return {"x": x, "invalid": True, "queries": len(points)}
            err = abs(got - want)
            if err > worst_error:
                worst_error = err
                worst = {"x": x, "expected": want, "observed": got,
                         "max_absolute_error": err, "queries": len(points)}
        return worst

    def holdout_score(self, candidate: Expr) -> Fit:
        return fit(candidate, tuple((x, self._target.evaluate(x)) for x in self.holdout_x))

    def reveal(self) -> dict:
        return {"task_id": self.task_id, "family": self.family,
                "target": self._target.serial(), "train_x": self.train_x,
                "challenge_x": self.challenge_x, "holdout_x": self.holdout_x}


def make_tasks(seed: int, n_train: int = 10, n_test: int = 6):
    """Independent fixed-seed task generation. Test family labels not exposed to search."""
    import random
    rng = random.Random(seed)
    x = Expr.var()
    x2 = expr("mul", x, x)
    x3 = expr("mul", x2, x)

    def create(idx, family):
        a = rng.choice((-2, -1, 1, 2))
        b = rng.choice((-2, -1, 0, 1, 2))
        c = rng.choice((-2, -1, 0, 1, 2))
        if family == "linear":
            target = expr("add", expr("mul", Expr.const(a), x), Expr.const(c))
        elif family == "quadratic":
            target = expr("add", expr("add", expr("mul", Expr.const(a), x2),
                                                   expr("mul", Expr.const(b), x)), Expr.const(c))
        else:
            target = expr("add", expr("add", expr("mul", Expr.const(a), x3),
                                                   expr("mul", Expr.const(b), x)), Expr.const(c))
        return SealedTask(f"T{idx:03d}", family, target)

    train = [create(i, "linear" if i % 2 == 0 else "quadratic") for i in range(n_train)]
    test = [create(n_train+i, "quadratic" if i % 2 == 0 else "cubic") for i in range(n_test)]
    return train, test
