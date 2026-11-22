"""Evaluator-owned polynomial identity and disjoint public task generation.

Exact coefficient tuples identify mathematical polynomials in the current
integer DSL, NOT equivalence of bounded-intermediate runtime behavior.
Do not expose keys, target ASTs or secret generator seeds to a learner.
This module is public development infrastructure, not an isolation boundary.
"""
from __future__ import annotations

from itertools import product
import random

from .core import Expr, SealedTask, expr

SPLIT_PROTOCOL = "USF-POLYNOMIAL-SPLIT-1"
PUBLIC_DEV3_SEEDS = (78301, 78302, 78303)
MAX_DEGREE = 64
MAX_COEFFICIENT = 2**256


def polynomial_key(root: Expr) -> tuple[int, ...]:
    """Canonical ascending coefficient tuple, bounded and fail-closed."""
    remaining = 100

    def clean(values):
        while len(values) > 1 and values[-1] == 0:
            values.pop()
        if len(values)-1 > MAX_DEGREE or any(abs(x) > MAX_COEFFICIENT for x in values):
            raise ValueError("polynomial identity exceeds configured bounds")
        return tuple(values)

    def visit(node, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 30 or type(node) is not Expr:
            raise ValueError("invalid or oversized expression")
        if node.op == "x":
            return (0, 1)
        if node.op == "const":
            return (node.value,)
        a = visit(node.args[0], depth+1)
        if node.op == "neg":
            return clean([-x for x in a])
        b = visit(node.args[1], depth+1)
        if node.op in ("add", "sub"):
            out = [0] * max(len(a), len(b))
            for i, x in enumerate(a):
                out[i] += x
            for i, y in enumerate(b):
                out[i] += y if node.op == "add" else -y
            return clean(out)
        if node.op != "mul":
            raise ValueError("unsupported polynomial operation")
        if len(a)+len(b)-2 > MAX_DEGREE:
            raise ValueError("polynomial degree exceeds bound")
        out = [0] * (len(a)+len(b)-1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i+j] += x*y
        return clean(out)

    return visit(root, 1)


def verify_semantic_splits(fit, validation, final):
    """Evaluator only. Reject duplicate IDs/functions within/across all splits.

    Returns counts, not target coefficient tuples or per-target hashes.
    Low-entropy hidden targets require independently held, salted/keyed
    commitments; this checker does not authenticate a split commitment.
    """
    seen_ids = set()
    seen_functions = {}
    counts = {}
    for label, tasks in (("fit", fit), ("validation", validation), ("final", final)):
        count = 0
        for task in tasks:
            count += 1
            if count > 32 or type(task) is not SealedTask:
                raise ValueError("invalid evaluator task set")
            if type(task.task_id) is not str or not task.task_id.strip() or task.task_id in seen_ids:
                raise ValueError("duplicate or invalid task identity")
            seen_ids.add(task.task_id)
            key = polynomial_key(task._target)
            if key in seen_functions:
                raise ValueError("semantic overlap: "+seen_functions[key]+" / "+label)
            seen_functions[key] = label
        if count == 0:
            raise ValueError("empty split")
        counts[label] = count
    return {
        "protocol": SPLIT_PROTOCOL, "counts": counts,
        "unique_polynomial_functions": len(seen_functions),
        "semantic_disjointness_checked": True,
        "independent_provenance_authenticated": False,
        "protected_evaluation_established": False,
    }


def make_semantic_splits(seed, n_fit=20, n_validation=6, n_final=6):
    """Public evaluator-side deterministic sample without replacement.

    Family schedule matches DEV-2's structure, but functions are distinct.
    Fixed polynomial coefficient pools; no performance-dependent selection.
    Do not substitute this generator into an already executed protocol.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("invalid seed")
    sizes = (n_fit, n_validation, n_final)
    if any(type(n) is not int or not 1 <= n <= 32 for n in sizes):
        raise ValueError("split sizes must be in [1,32]")
    slopes = (-2,-1,1,2)
    offsets = (-2,-1,0,1,2)
    pools = {
        "linear": [(c,a) for a,c in product(slopes,offsets)],
        "quadratic": [(c,b,a) for a,b,c in product(slopes,offsets,offsets)],
        "cubic": [(c,b,0,a) for a,b,c in product(slopes,offsets,offsets)],
    }
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    schedule = (
        ["linear" if i%2 == 0 else "quadratic" for i in range(n_fit)],
        ["linear" if (n_fit+i)%2 == 0 else "quadratic" for i in range(n_validation)],
        ["quadratic" if i%2 == 0 else "cubic" for i in range(n_final)],
    )
    all_families = [family for group in schedule for family in group]
    if any(all_families.count(family) > len(pool) for family,pool in pools.items()):
        raise ValueError("insufficient distinct functions for requested split")
    x = Expr.var()
    x2 = expr("mul",x,x)
    x3 = expr("mul",x2,x)
    splits = []
    idx = 0
    for group in schedule:
        tasks = []
        for family in group:
            coeff = pools[family].pop()
            c,b = coeff[0],coeff[1]
            if family == "linear":
                target = expr("add",expr("mul",Expr.const(b),x),Expr.const(c))
            else:
                a = coeff[-1]
                power = x2 if family == "quadratic" else x3
                target = expr("add",expr("add",expr("mul",Expr.const(a),power),
                                         expr("mul",Expr.const(b),x)),Expr.const(c))
            if polynomial_key(target) != coeff:
                raise RuntimeError("generator semantic invariant failed")
            tasks.append(SealedTask(f"T{idx:03d}",family,target))
            idx += 1
        splits.append(tasks)
    verify_semantic_splits(*splits)
    return tuple(splits)
