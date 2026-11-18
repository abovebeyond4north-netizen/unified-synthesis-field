"""Prospective public DEV-3 protocol; no experiment runs on import.

Use preflight_dev3() to obtain the exact source/protocol review commitment.
Execution requires that separately retained commitment; a digest is NOT
approval or authentication. Never substitute these splits into DEV-2.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter_ns

from .known_composition import (
    FIT_TASKS, VALIDATION_TASKS, FINAL_TASKS, PROPOSAL_CAP, ATTEMPT_CAP,
    VALIDATION_SEED_OFFSETS, protocol_descriptor,
)
from .macro_final import freeze_validation, evaluate_frozen, _read_committed
from .macro_utility import digest
from .macro_validation import run_validation, _source_hashes
from .semantic_splits import (
    SPLIT_PROTOCOL, PUBLIC_DEV3_SEEDS, make_semantic_splits, verify_semantic_splits,
)

PROTOCOL = "USF-MACRO-UTILITY-PUBLIC-DEV-3"


def dev3_descriptor():
    return {
        **protocol_descriptor(),
        "protocol": PROTOCOL,
        "development_seeds": list(PUBLIC_DEV3_SEEDS),
        "task_generation": "make_semantic_splits(seed,20,6,6)",
        "split_protocol": SPLIT_PROTOCOL,
        "semantic_identity": "exact_integer_polynomial_coefficients",
        "semantic_disjointness": "within_and_across_splits_per_seed",
        "seed_selection": "DEV-2 seeds plus 100, fixed before DEV-3 search",
        "cross_seed_overlap_allowed": True,
        "shared_learning_across_development_seeds": False,
        "validation_repeats_are_independent_tasks": False,
        "prior_protocol": "DEV-2 retained as semantically contaminated engineering control",
    }


def _sources():
    root = Path(__file__).resolve().parent
    return {**_source_hashes(), **{
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("macro_final.py", "semantic_splits.py", "macro_dev3.py")
    }}


def _checked_splits(seed):
    splits = make_semantic_splits(seed, FIT_TASKS, VALIDATION_TASKS, FINAL_TASKS)
    check = verify_semantic_splits(*splits)
    if check["counts"] != {"fit": FIT_TASKS, "validation": VALIDATION_TASKS, "final": FINAL_TASKS}:
        raise ValueError("split count mismatch")
    if check["unique_polynomial_functions"] != FIT_TASKS + VALIDATION_TASKS + FINAL_TASKS:
        raise ValueError("nonunique polynomial functions")
    return splits, check


def preflight_dev3():
    """No search, no scores, no target values or per-target digests returned."""
    descriptor = dev3_descriptor()
    checks = []
    for seed in PUBLIC_DEV3_SEEDS:
        _, check = _checked_splits(seed)
        checks.append({"seed": seed, **check})
    body = {
        "protocol": descriptor,
        "protocol_sha256": digest(descriptor),
        "source_hashes": _sources(),
        "split_checks": checks,
        "search_performed": False,
        "independent_review_established": False,
        "milestone_claim_eligible": False,
    }
    return {**body, "preflight_sha256": digest(body)}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _snapshot(splits):
    # Evaluator-side local identity only. Never expose target digests; even
    # public-style polynomial pools are too small for unsalted secrecy.
    return digest([[task.reveal() for task in group] for group in splits])


def run_dev3(seed, directory, *, expected_preflight_sha256):
    """Run one fixed seed only against an explicitly retained preflight digest.

    This guard is not an authorization service. A reviewer must separately
    approve the exact version before an operator invokes this public runner.
    Final task objects are kept out of the validation runner, not isolated
    from the Python process. DEV-2 files/runner are left unchanged.
    """
    _require(type(seed) is int and seed in PUBLIC_DEV3_SEEDS, "unknown DEV-3 seed")
    started = perf_counter_ns()
    preflight = preflight_dev3()
    _require(type(expected_preflight_sha256) is str and
             preflight["preflight_sha256"] == expected_preflight_sha256,
             "DEV-3 preflight commitment mismatch")
    directory = Path(directory)
    frozen = directory / f"dev3-{seed}-validation.json"
    final_path = directory / f"dev3-{seed}-final.json"
    _require(not frozen.exists() and not final_path.exists(), "existing DEV-3 artifact")
    splits, check = _checked_splits(seed)
    fit, validation, final = splits
    snapshot = _snapshot(splits)
    final_ids = [t.task_id for t in final]
    preflight_ns = perf_counter_ns() - started

    report = run_validation(fit, validation, final_ids, seed=seed, budget=PROPOSAL_CAP,
                            validation_offsets=VALIDATION_SEED_OFFSETS)
    _require(snapshot == _snapshot(splits), "task split changed during validation")
    _require(preflight == preflight_dev3(), "source or protocol drift")
    manifest = report["manifest"]
    _require(report["development_seed"] == seed and
             manifest["fit_ids"] == [t.task_id for t in fit] and
             manifest["validation_ids"] == [t.task_id for t in validation] and
             manifest["test_ids"] == final_ids and
             manifest["seeds"] == [seed + off for off in VALIDATION_SEED_OFFSETS] and
             manifest["proposal_cap"] == PROPOSAL_CAP and
             manifest["attempt_cap"] == ATTEMPT_CAP, "validation manifest mismatch")
    report["protocol_sha256"] = preflight["protocol_sha256"]
    report["dev3_preflight"] = preflight
    report["semantic_split_receipt"] = check
    report["dev3_preflight_ns"] = preflight_ns
    directory.mkdir(parents=True, exist_ok=True)
    commitment = freeze_validation(report, frozen)
    envelope = _read_committed(frozen, commitment)
    _require(envelope["validation_report"]["dev3_preflight"] == preflight,
             "persisted preflight mismatch")
    _require(snapshot == _snapshot(splits) and preflight == preflight_dev3(),
             "task or protocol drift before final phase")
    result = evaluate_frozen(frozen, commitment, final, final_path)
    _require(snapshot == _snapshot(splits) and preflight == preflight_dev3(),
             "task or protocol drift during final phase")
    return report, result
