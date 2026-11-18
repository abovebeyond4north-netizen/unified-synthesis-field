"""Commit validation evidence before evaluating final tasks.

Local SHA-256 commitments and exclusive files detect accidental changes when
the expected digest is retained separately. They are not signatures, trusted
timestamps, a hostile-process security boundary, or independent verification.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import hmac
import json
import os
from pathlib import Path
from time import perf_counter_ns

from .core import make_tasks
from .known_composition import (
    BASE_ACTIONS, DEVELOPMENT_SEEDS, FIT_TASKS, VALIDATION_TASKS, FINAL_TASKS,
    PROPOSAL_CAP, VALIDATION_SEED_OFFSETS, FINAL_SEED_OFFSET,
    comparator_digests, protocol_descriptor,
)
from .macro_induction import induce_root_macro
from .macro_utility import ARITY, ARMS, decide_utility, digest
from .macro_validation import _source_hashes, run_dev2_validation
from .search import search

SCHEMA = "USF-MACRO-FINAL-FREEZE-1"
MAX_BYTES = 8 * 1024 * 1024


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _unique(pairs):
    out = {}
    for key, value in pairs:
        _require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def _reject_constant(value):
    raise ValueError("nonfinite JSON value: " + value)


def _encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _decode(raw):
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=_reject_constant)


def _hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _durable_create(path, payload):
    raw = _encode(payload)
    _require(len(raw) <= MAX_BYTES, "artifact too large")
    # Refuse to replace an existing receipt, even if its contents match.
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return _hash_bytes(raw)


def _read_committed(path, expected):
    _require(type(expected) is str and len(expected) == 64, "missing commitment")
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    _require(len(raw) <= MAX_BYTES, "artifact too large")
    _require(hmac.compare_digest(_hash_bytes(raw), expected), "commitment mismatch")
    decoded = _decode(raw)
    _require(_encode(decoded) == raw, "noncanonical artifact")
    return decoded


def _final_code_hash():
    return _hash_bytes(Path(__file__).read_bytes())


def _verified_macro(report):
    """Recompute the gate and induction from local, self-reported evidence."""
    _require(type(report) is dict, "invalid report")
    try:
        m = report["manifest"]
        c = report["candidate"]
        rows = report["validation_rows"]
        training = report["training"]
        _require(report["final_search_performed"] is False, "final phase already executed")
        _require(report["source_hashes"] == _source_hashes(), "source drift")
        _require(m["architecture_sha256"] == digest(report["source_hashes"]), "architecture commitment")
        _require(m["grammar_sha256"] == digest({"arity": ARITY, "actions": list(BASE_ACTIONS)}),
                 "grammar commitment")
        _require(m["comparators"] == comparator_digests(), "comparator drift")
        expected_decision = decide_utility(m, c, rows)
        _require(expected_decision["reason"] != "REJECT_INVALID_EVIDENCE", "invalid validation evidence")
        _require(report["decision"] == expected_decision, "decision does not recompute")
        # Auxiliary score receipts must agree with the binary decision inputs.
        scores = report["validation_scores"]
        _require(type(scores) is list and len(scores) == len(rows), "score receipt count")
        indexed = {(r["seed"], r["task_id"], r["arm"]): r for r in rows}
        seen = set()
        for score in scores:
            key = (score["seed"], score["task_id"], score["arm"])
            fraction = score["exact_fraction"]
            _require(key in indexed and key not in seen, "score receipt pairing")
            _require(type(fraction) in (float, int) and 0 <= fraction <= 1 and
                     int(fraction == 1) == indexed[key]["exact_success"],
                     "score receipt outcome mismatch")
            _require(type(score["score_ns"]) is int and score["score_ns"] >= 0,
                     "invalid scoring duration")
            seen.add(key)

        _require(type(training) is list and len(training) == len(m["fit_ids"]), "incomplete fit receipts")
        seed = report["development_seed"]
        _require(type(seed) is int and 0 <= seed < 2**63 - 1_000_000, "invalid development seed")
        observations = []
        for i, row in enumerate(training):
            _require(row["task_id"] == m["fit_ids"][i], "fit order or identity mismatch")
            _require(row["search_seed"] == seed + i * 101 + 7, "fit seed mismatch")
            _require(type(row["challenge_survived"]) is bool, "invalid survival flag")
            p, a = row["proposals"], row["attempts"]
            _require(type(p) is int and 6 <= p <= m["proposal_cap"] and
                     type(a) is int and p - 6 <= a <= m["attempt_cap"], "invalid fit work")
            if row["challenge_survived"]:
                observations.append((row["task_id"], row["selected_ast"]))
        induced = induce_root_macro(observations)
        recovered = _decode(_encode(asdict(induced))) if induced is not None else None
        _require(recovered == c, "candidate does not recompute from fit receipts")
        _require(report["surviving_training_observations"] == len(observations), "fit count mismatch")
        _require(report["validation_skipped_no_macro"] is (c is None), "validation phase mismatch")
        return induced if expected_decision["admitted"] else None
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed validation report") from exc


def freeze_validation(report, path):
    """Persist the full validation evidence and return a separate byte digest."""
    # Normalize tuples without accepting nonfinite values or aliases.
    snapshot = _decode(_encode(report))
    _verified_macro(snapshot)
    envelope = {
        "schema": SCHEMA, "final_code_sha256": _final_code_hash(),
        "final_seed_offset": FINAL_SEED_OFFSET, "validation_report": snapshot,
    }
    return _durable_create(path, envelope)


def _load_frozen(path, expected):
    envelope = _read_committed(path, expected)
    _require(type(envelope) is dict and set(envelope) == {
        "schema", "final_code_sha256", "final_seed_offset", "validation_report"
    }, "invalid envelope")
    _require(envelope["schema"] == SCHEMA and
             envelope["final_code_sha256"] == _final_code_hash() and
             envelope["final_seed_offset"] == FINAL_SEED_OFFSET, "final protocol drift")
    report = envelope["validation_report"]
    return report, _verified_macro(report)


def evaluate_frozen(path, expected, final_tasks, output_path):
    """Evaluate primitive, known, and gated policies; never reuse a rejected macro.

    Same-process public development only. Every final search completes before
    final scoring. This cannot enforce blindness against a malicious caller.
    """
    start = perf_counter_ns()
    _require(not Path(output_path).exists(), "final receipt already exists")
    report, admitted_macro = _load_frozen(path, expected)
    tasks = tuple(final_tasks)
    m = report["manifest"]
    _require([t.task_id for t in tasks] == m["test_ids"], "final task IDs/order mismatch")
    setup_ns = perf_counter_ns() - start
    budget = m["proposal_cap"]
    seed = report["development_seed"] + FINAL_SEED_OFFSET
    pending = []
    for i, task in enumerate(tasks):
        for arm in ARMS:
            before = perf_counter_ns()
            item = search(task, budget, seed + i * 101, "evolution",
                          macro=admitted_macro if arm == "macro" else None,
                          known_composition=arm == "known")
            pending.append((task, arm, item, perf_counter_ns() - before))
    rows = []
    for task, arm, item, elapsed in pending:
        before = perf_counter_ns()
        score = task.holdout_score(item.selected.candidate)
        score_ns = perf_counter_ns() - before
        rows.append({
            "task_id": task.task_id, "arm": arm,
            "selected_ast": item.selected.candidate.serial(),
            "exact_success": int(score.exact_fraction == 1),
            "exact_fraction": score.exact_fraction,
            "proposals": item.proposals, "attempts": item.attempts,
            "macro_attempts": item.macro_attempts,
            "macro_proposals": item.macro_proposals,
            "search_ns": elapsed, "score_ns": score_ns,
        })
    # Recheck bytes, source files and the gate before recording a final result.
    _, again = _load_frozen(path, expected)
    _require(again == admitted_macro, "admitted representation changed")
    totals = {
        arm: {key: sum(r[key] for r in rows if r["arm"] == arm)
              for key in ("exact_success", "proposals", "attempts",
                          "macro_attempts", "macro_proposals", "search_ns", "score_ns")}
        for arm in ARMS
    }
    result = {
        "schema": SCHEMA, "development_seed": report["development_seed"],
        "validation_commitment": expected, "decision_sha256": report["decision"]["decision_sha256"],
        "decision_reason": report["decision"]["reason"],
        "macro_admitted": admitted_macro is not None,
        "macro_arm_fallback": admitted_macro is None,
        "rows": rows, "totals": totals, "setup_ns": setup_ns,
        "elapsed_through_evaluation_ns": perf_counter_ns() - start,
        "independent_provenance_authenticated": False,
        "matched_actual_compute_established": False,
        "milestone_claim_eligible": False,
    }
    _durable_create(output_path, result)
    return result


def run_dev2(seed, directory):
    """Execute one prospectively declared public DEV-2 seed, retaining both files."""
    _require(type(seed) is int and seed in DEVELOPMENT_SEEDS, "unknown DEV-2 seed")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frozen_path = directory / f"dev2-{seed}-validation.json"
    final_path = directory / f"dev2-{seed}-final.json"
    _require(not frozen_path.exists() and not final_path.exists(), "existing DEV-2 artifact")
    report = run_dev2_validation(seed)
    _require(report["protocol_sha256"] == digest(protocol_descriptor()), "DEV-2 protocol mismatch")
    expected = freeze_validation(report, frozen_path)
    # Regenerate final task objects for evaluation after the commitment is written.
    _, final = make_tasks(seed, FIT_TASKS + VALIDATION_TASKS, FINAL_TASKS)
    result = evaluate_frozen(frozen_path, expected, final, final_path)
    return report, result
