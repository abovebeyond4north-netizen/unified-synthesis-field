"""Minimal controlled symbolic search: random, evolutionary, and adaptive policies.

This is NOT GATv2, actor-critic, or MCTS. It provides their experimental substrate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .core import Expr, Fit, SealedTask, expr, fit
from .ledger import Ledger

ACTIONS = ("add_x", "sub_x", "mul_x", "add_one", "sub_one", "mul_two", "negate", "square")
INITIAL = (Expr.var(), Expr.const(0), Expr.const(1), Expr.const(-1), Expr.const(2), Expr.const(-2))


def mutate(h: Expr, action: str) -> Expr:
    x = Expr.var()
    one = Expr.const(1)
    if action == "add_x":
        return expr("add", h, x)
    if action == "sub_x":
        return expr("sub", h, x)
    if action == "mul_x":
        return expr("mul", h, x)
    if action == "add_one":
        return expr("add", h, one)
    if action == "sub_one":
        return expr("sub", h, one)
    if action == "mul_two":
        return expr("mul", h, Expr.const(2))
    if action == "negate":
        return expr("neg", h)
    if action == "square":
        return expr("mul", h, h)
    raise ValueError("Unknown action")


class AdaptivePolicy:
    """Across-training-task operator preferences; frozen during held-out evaluation."""
    def __init__(self):
        self.counts = {x: 0 for x in ACTIONS}
        self.gains = {x: 0.0 for x in ACTIONS}

    def select(self, rng: random.Random, adaptive: bool):
        if not adaptive:
            return rng.choice(ACTIONS)
        # Weak exploration prevents permanent elimination of an action.
        weights = [0.25 + max(0.0, self.gains[a] / max(1, self.counts[a])) for a in ACTIONS]
        return rng.choices(ACTIONS, weights=weights, k=1)[0]

    def update(self, action: str, gain_bits: float):
        self.counts[action] += 1
        self.gains[action] += min(10.0, max(0.0, gain_bits))

    def snapshot(self):
        return {"counts": self.counts.copy(), "gains": self.gains.copy()}


@dataclass(frozen=True)
class Scored:
    candidate: Expr
    result: Fit
    counterexample: dict | None
    candidate_id: str
    parent_id: str | None

    @property
    def survived(self) -> bool:
        return self.result.exact_fraction == 1.0 and self.counterexample is None


@dataclass(frozen=True)
class SearchResult:
    selected: Scored
    proposals: int
    attempts: int
    survived_count: int
    mode: str


def _quality(s: Scored) -> float:
    # Challenge successes are used for *selection*, but not treated as proofs.
    return (10000 if s.survived else 0) + s.result.compression_gain_bits


def search(task: SealedTask, budget: int, seed: int, mode: str,
           policy: AdaptivePolicy | None = None, train: bool = False,
           ledger: Ledger | None = None, run_id: str = "", label: str | None = None) -> SearchResult:
    if mode not in ("random", "evolution", "adaptive"):
        raise ValueError("Unknown comparator")
    if not (6 <= budget <= 20000):
        raise ValueError("budget must be in [6, 20000]")
    if mode == "adaptive" and policy is None:
        raise ValueError("Adaptive mode requires a policy")
    label = label or mode
    rng = random.Random(seed)
    samples = task.training_samples()
    pool: list[Scored] = []
    seen: set[str] = set()
    attempts = 0

    def register(ast: Expr, parent: Scored | None, action: str) -> Scored:
        ident = f"{label}:{task.task_id}:{len(pool)+1:05d}"
        metrics = fit(ast, samples)
        ce = task.falsify(ast) if metrics.exact_fraction == 1.0 else {"skipped": "TRAIN_MISMATCH"}
        item = Scored(ast, metrics, ce, ident, parent.candidate_id if parent else None)
        pool.append(item)
        seen.add(ast.key())
        if ledger is not None:
            ledger.append(run_id, "candidate", {
                "task_id": task.task_id, "mode": label, "phase": "train" if train else "test",
                "generation": len(pool)-1, "expression": str(ast), "ast": ast.serial(),
                "action": action, "mdl_bits": metrics.mdl_bits, "compression_gain_bits": metrics.compression_gain_bits,
                "train_exact": metrics.exact_fraction, "train_mae": metrics.mae,
                "falsification_survived": item.survived,
                "counterexample": ce, "logos_status": "NOT_SUBMITTED",
                "axiom_audit_pass": False,
            }, ident, parent.candidate_id if parent else None)
        return item

    for initial in INITIAL:
        register(initial, None, "INITIAL")

    while len(pool) < budget and attempts < budget * 10:
        attempts += 1
        if mode == "random":
            parent = rng.choice(pool)
        else:
            top = sorted(pool, key=_quality, reverse=True)[:min(12, len(pool))]
            parent = rng.choice(top if rng.random() < 0.85 else pool)
        action = policy.select(rng, True) if mode == "adaptive" else rng.choice(ACTIONS)
        candidate = mutate(parent.candidate, action)
        if candidate.size() > 23 or candidate.depth() > 9 or candidate.key() in seen:
            continue
        current_best = max(pool, key=_quality)
        item = register(candidate, parent, action)
        if train and policy is not None and mode == "adaptive":
            # Credit only independent training improvements; never sees held-out data.
            policy.update(action, _quality(item) - _quality(current_best))

    survivors = [p for p in pool if p.survived]
    # Among surviving candidates, prefer smallest full two-part code length.
    selected = min(survivors, key=lambda p: (p.result.mdl_bits, p.candidate.size(), p.candidate.key())) if survivors else max(pool, key=_quality)
    return SearchResult(selected, len(pool), attempts, len(survivors), label)
