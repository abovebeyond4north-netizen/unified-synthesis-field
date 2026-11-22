"""Experimental root-template induction from observed training ASTs only.

One-level macros, exact syntactic expansion, node-count proxy (not MDL bits).
No evaluator access, policy mutation, formal proof or milestone admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json

ARITY = {"x": 0, "const": 0, "neg": 1, "add": 2, "sub": 2, "mul": 2}


def _tree(raw, depth=1, remaining=None):
    if remaining is None:
        remaining = [100]
    remaining[0] -= 1
    if remaining[0] < 0:
        raise ValueError("AST too large")
    if depth > 30 or type(raw) not in (list, tuple) or not raw or type(raw[0]) is not str:
        raise ValueError("invalid AST")
    op = raw[0]
    if op not in ARITY:
        raise ValueError("unknown opcode")
    if op == "const":
        if len(raw) != 2 or type(raw[1]) is not int or abs(raw[1]) > 1000:
            raise ValueError("invalid constant")
        return tuple(raw)
    if len(raw) != 1 + ARITY[op]:
        raise ValueError("invalid arity")
    return (op, *(_tree(c, depth + 1, remaining) for c in raw[1:]))


def _children(t):
    return () if t[0] in ("const", "x", "hole") else t[1:]


def _size(t):
    return 1 + sum(_size(c) for c in _children(t))


def _pattern(a, b):
    holes = {}
    def visit(x, y):
        if x == y:
            return x
        if x[0] == y[0] and x[0] not in ("const", "x"):
            return (x[0], *(visit(u, v) for u, v in zip(x[1:], y[1:])))
        key = (x, y)
        if key not in holes:
            holes[key] = len(holes)
        return ("hole", holes[key])
    return visit(a, b), len(holes)


def _match(pattern, tree, bindings):
    if pattern[0] == "hole":
        i = pattern[1]
        if i in bindings:
            return bindings[i] == tree
        bindings[i] = tree
        return True
    if pattern[0] != tree[0]:
        return False
    if pattern[0] in ("const", "x"):
        return pattern == tree
    return all(_match(p, t, bindings) for p, t in zip(pattern[1:], tree[1:]))


def _ops(t):
    return int(t[0] in ("neg", "add", "sub", "mul")) + sum(_ops(c) for c in _children(t))


@dataclass(frozen=True)
class MacroCandidate:
    pattern: tuple
    arity: int
    source_ids: tuple[str, ...]
    training_node_savings: int

    def arguments(self, raw):
        tree = _tree(raw)
        if _size(tree) > 100:
            raise ValueError("AST too large")
        bindings = {}
        if not _match(self.pattern, tree, bindings):
            return None
        return tuple(bindings[i] for i in range(self.arity))

    def expand(self, arguments):
        args = tuple(_tree(a) for a in arguments)
        if len(args) != self.arity or any(_size(a) > 100 for a in args):
            raise ValueError("invalid macro arguments")
        def visit(t):
            if t[0] == "hole":
                return args[t[1]]
            if t[0] in ("const", "x"):
                return t
            return (t[0], *(visit(c) for c in t[1:]))
        result = _tree(visit(self.pattern))
        if _size(result) > 100:
            raise ValueError("expanded AST too large")
        return result


def induce_root_macro(observations):
    """Return best compressive candidate, or None. Input: <=32 (source_id, AST).

    Caller must supply training-only observations; source IDs are not authenticated.
    Cost = template nodes + one definition node + one call node per use + arguments.
    Only distinct expressions from distinct sources count as supporting examples.
    """
    samples = []
    ids = set()
    for ident, raw in observations:
        if len(samples) >= 32 or type(ident) is not str or not ident.strip() or ident in ids:
            raise ValueError("invalid or duplicate source; maximum 32")
        t = _tree(raw)
        if _size(t) > 100:
            raise ValueError("AST too large")
        ids.add(ident)
        samples.append((ident, t))
    samples.sort()
    patterns = set()
    for (_, a), (_, b) in combinations(samples, 2):
        if a != b:
            p, n = _pattern(a, b)
            if 1 <= n <= 4 and _ops(p) >= 2:
                patterns.add((p, n))
    candidates = []
    for p, n in patterns:
        uses = {}
        for ident, t in samples:
            bindings = {}
            if _match(p, t, bindings) and t not in uses:
                args = tuple(bindings[i] for i in range(n))
                gain = _size(t) - 1 - sum(_size(a) for a in args)
                if gain > 0:
                    uses[t] = (ident, gain)
        saving = sum(g for _, g in uses.values()) - _size(p) - 1
        if len(uses) >= 2 and saving > 0:
            candidates.append(MacroCandidate(p, n, tuple(sorted(i for i, _ in uses.values())), saving))
    return min(candidates, key=lambda c: (-c.training_node_savings, json.dumps(c.pattern))) if candidates else None
