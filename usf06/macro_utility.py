"""Pure development-only representation admission: no task/evaluator access.

This gate validates self-reported receipts. It does not authenticate a split,
prove a macro, establish compute equivalence, or certify scientific acceptance.
All inputs are JSON-shaped records; callers retain the raw receipts.
"""
from __future__ import annotations

import hashlib
import json
import re

RULE = "USF-MACRO-UTILITY-1"
ARMS = ("primitive", "known", "macro")
MANIFEST_KEYS = {
    "rule", "fit_ids", "validation_ids", "test_ids", "seeds",
    "proposal_cap", "attempt_cap", "architecture_sha256",
    "grammar_sha256", "comparators",
}
ROW_KEYS = {
    "seed", "task_id", "arm", "manifest_sha256", "candidate_sha256",
    "comparator_sha256", "proposal_cap", "attempt_cap", "exact_success",
    "proposals", "attempts", "macro_attempts", "macro_proposals", "search_ns",
}
HEX = re.compile(r"[0-9a-f]{64}")
ARITY = {"x": 0, "const": 0, "neg": 1, "add": 2, "sub": 2, "mul": 2}


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _record(value, keys, label):
    _require(type(value) is dict and set(value) == keys, label + ": fields")


def _int(value, lo, hi, label):
    _require(type(value) is int and lo <= value <= hi, label)
    return value


def _hash(value, label):
    _require(type(value) is str and HEX.fullmatch(value) is not None, label)


def _ids(values, label):
    _require(type(values) is list and 1 <= len(values) <= 32, label)
    _require(all(type(v) is str and 0 < len(v) <= 128 and v.strip() == v
                 for v in values), label)
    _require(len(set(values)) == len(values), label + ": duplicates")
    return set(values)


def _manifest(m):
    _record(m, MANIFEST_KEYS, "manifest")
    _require(m["rule"] == RULE, "rule version")
    fit = _ids(m["fit_ids"], "fit IDs")
    val = _ids(m["validation_ids"], "validation IDs")
    test = _ids(m["test_ids"], "test IDs")
    _require(not (fit & val or fit & test or val & test), "split overlap")
    seeds = m["seeds"]
    _require(type(seeds) is list and 1 <= len(seeds) <= 32, "seeds")
    for s in seeds:
        _int(s, 0, 2**63-1, "seed")
    _require(len(set(seeds)) == len(seeds), "duplicate seeds")
    cap = _int(m["proposal_cap"], 6, 2000, "proposal cap")
    _int(m["attempt_cap"], 1, 20000, "attempt cap")
    _require(m["attempt_cap"] == cap * 10, "search cap contract")
    for key in ("architecture_sha256", "grammar_sha256"):
        _hash(m[key], key)
    _record(m["comparators"], set(ARMS), "comparators")
    for value in m["comparators"].values():
        _hash(value, "comparator digest")
    _require(len(set(m["comparators"].values())) == len(ARMS), "comparator identity")
    return fit, val


def _candidate(c, fit):
    _record(c, {"pattern", "arity", "source_ids", "training_node_savings"}, "candidate")
    arity = _int(c["arity"], 1, 4, "macro arity")
    sources = _ids(c["source_ids"], "source IDs")
    _require(len(sources) >= 2 and sources <= fit, "non-fit or insufficient sources")
    _int(c["training_node_savings"], 1, 3200, "training node savings")
    holes = set()
    nodes = 0
    ops = 0

    def visit(t, depth):
        nonlocal nodes, ops
        nodes += 1
        _require(nodes <= 100 and depth <= 30, "pattern bounds")
        _require(type(t) is list and bool(t) and type(t[0]) is str, "pattern node")
        op = t[0]
        if op in ("hole", "const"):
            _require(len(t) == 2, "leaf arity")
            if op == "hole":
                holes.add(_int(t[1], 0, arity-1, "hole index"))
            else:
                _int(t[1], -1000, 1000, "constant")
            return
        _require(op in ARITY and len(t) == 1 + ARITY[op], "operator arity")
        if op != "x":
            ops += 1
        for child in t[1:]:
            visit(child, depth + 1)

    visit(c["pattern"], 1)
    _require(holes == set(range(arity)) and ops >= 2, "pattern structure")


def _row(r, m, mh, ch, val):
    _record(r, ROW_KEYS, "validation row")
    seed = _int(r["seed"], 0, 2**63-1, "row seed")
    _require(seed in m["seeds"], "unknown seed")
    task, arm = r["task_id"], r["arm"]
    _require(type(task) is str and task in val, "non-validation task")
    _require(type(arm) is str and arm in ARMS, "unknown arm")
    _require(r["manifest_sha256"] == mh and r["candidate_sha256"] == ch
             and r["comparator_sha256"] == m["comparators"][arm], "commitment mismatch")
    for cap in ("proposal_cap", "attempt_cap"):
        _require(type(r[cap]) is int and r[cap] == m[cap], "cap mismatch")
    exact = r["exact_success"]
    _require(type(exact) in (int, float) and exact in (0, 1), "binary exact success")
    p = _int(r["proposals"], 6, m["proposal_cap"], "proposals")
    a = _int(r["attempts"], max(0, p-6), m["attempt_cap"], "attempts")
    ma = _int(r["macro_attempts"], 0, a, "macro attempts")
    mp = _int(r["macro_proposals"], 0, min(ma, p-6), "macro proposals")
    _require(p-6-mp <= a-ma, "primitive/macro accounting")
    _require(arm != "primitive" or (ma == 0 and mp == 0), "primitive macro usage")
    _int(r["search_ns"], 0, 2**63-1, "search duration")
    return seed, task, arm


def decide_utility(manifest, candidate, rows):
    """Return a deterministic decision and commitment, never a science claim.

    Ties/losses against either arm reject. Both successful and rejected raw
    receipts belong in caller-owned storage; the result contains their digest.
    No macro is a valid negative outcome only with no validation rows.
    """
    mh = ch = rh = None
    totals = None
    reason = None
    try:
        fit, val = _manifest(manifest)
        mh = digest(manifest)
        _require(type(rows) is list, "receipts must be a list")
        if candidate is None:
            _require(not rows, "receipts without macro")
            reason = "REJECT_NO_MACRO"
            rh = digest([])
        else:
            _candidate(candidate, fit)
            ch = digest(candidate)
            expected = {(s, t, a) for s in manifest["seeds"] for t in val for a in ARMS}
            _require(len(rows) == len(expected), "incomplete or excess receipts")
            indexed = {}
            for r in rows:
                key = _row(r, manifest, mh, ch, val)
                _require(key not in indexed, "duplicate validation receipt")
                indexed[key] = r
            _require(set(indexed) == expected, "incomplete pairing")
            rh = digest([indexed[k] for k in sorted(indexed)])
            totals = {
                arm: {key: sum(int(r[key]) for r in indexed.values() if r["arm"] == arm)
                      for key in ("exact_success", "proposals", "attempts",
                                  "macro_attempts", "macro_proposals", "search_ns")}
                for arm in ARMS
            }
            if totals["macro"]["macro_proposals"] == 0:
                reason = "REJECT_UNUSED"
            elif any(totals["macro"]["exact_success"] <= totals[a]["exact_success"]
                     for a in ("primitive", "known")):
                reason = "REJECT_NONPOSITIVE_UTILITY"
            else:
                reason = "ADMIT_DEVELOPMENT_ONLY"
    except (ValueError, TypeError, OverflowError, RecursionError):
        # No partial totals or commitment to malformed receipts is trusted.
        reason = "REJECT_INVALID_EVIDENCE"
        totals = rh = None
    result = {
        "rule": RULE, "reason": reason,
        "admitted": reason == "ADMIT_DEVELOPMENT_ONLY",
        "manifest_sha256": mh, "candidate_sha256": ch, "receipts_sha256": rh,
        "validation_totals": totals,
        "independent_provenance_authenticated": False,
        "matched_actual_compute_established": False,
        "milestone_claim_eligible": False,
    }
    return {**result, "decision_sha256": digest(result)}
