"""Fit-only induction and validation-only paired macro admission receipts.

No final task objects are accepted by run_validation. No final search is
implemented here. Public task IDs and local hashes are not authentication,
and the same-process evaluator is not a secrecy boundary.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

from .core import make_tasks
from .known_composition import (
    BASE_ACTIONS, DEVELOPMENT_SEEDS, FIT_TASKS, VALIDATION_TASKS, FINAL_TASKS,
    PROPOSAL_CAP, VALIDATION_SEED_OFFSETS, comparator_digests, protocol_descriptor,
)
from .macro_induction import induce_root_macro
from .macro_utility import ARITY, ARMS, RULE, decide_utility, digest
from .search import search


def _source_hashes():
    root = Path(__file__).resolve().parent
    names = ("core.py", "ledger.py", "search.py", "macro_induction.py",
             "macro_utility.py", "known_composition.py", "macro_validation.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names}


def run_validation(fit_tasks, validation_tasks, final_task_ids, *,
                   seed, budget, validation_offsets=VALIDATION_SEED_OFFSETS):
    """Generic development runner; final split is identities only.

    Each development seed induces its own macro. Receipt seed is the phase
    seed (seed + offset), and the search RNG adds validation index * 101.
    Time receipts measure local elapsed intervals, not matched physical compute.
    """
    start = perf_counter_ns()
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid development seed")
    if type(budget) is not int or not 6 <= budget <= 2000:
        raise ValueError("invalid budget")
    if type(validation_offsets) not in (tuple, list) or not 1 <= len(validation_offsets) <= 32:
        raise ValueError("invalid validation offsets")
    if any(type(v) is not int or v <= 0 for v in validation_offsets):
        raise ValueError("invalid validation offset")
    fit_tasks, validation_tasks = tuple(fit_tasks), tuple(validation_tasks)
    if type(final_task_ids) not in (list, tuple):
        raise ValueError("final task IDs must be list or tuple of strings")
    sources = _source_hashes()
    manifest = {
        "rule": RULE,
        "fit_ids": [t.task_id for t in fit_tasks],
        "validation_ids": [t.task_id for t in validation_tasks],
        "test_ids": list(final_task_ids),
        "seeds": [seed + offset for offset in validation_offsets],
        "proposal_cap": budget,
        "attempt_cap": budget * 10,
        "architecture_sha256": digest(sources),
        "grammar_sha256": digest({"arity": ARITY, "actions": list(BASE_ACTIONS)}),
        "comparators": comparator_digests(),
    }
    # Validate the complete split/cap contract before doing any search.
    initial = decide_utility(manifest, None, [])
    if initial["reason"] != "REJECT_NO_MACRO":
        raise ValueError("invalid manifest or overlapping split")
    mh = digest(manifest)
    setup_ns = perf_counter_ns() - start

    started = perf_counter_ns()
    training = []
    observations = []
    for i, task in enumerate(fit_tasks):
        before = perf_counter_ns()
        result = search(task, budget, seed + i * 101 + 7, "evolution", train=True)
        elapsed = perf_counter_ns() - before
        ast = result.selected.candidate.serial()
        training.append({
            "task_id": task.task_id, "search_seed": seed + i * 101 + 7,
            "selected_ast": ast, "challenge_survived": result.selected.survived,
            "proposals": result.proposals, "attempts": result.attempts,
            "search_ns": elapsed,
        })
        if result.selected.survived:
            observations.append((task.task_id, ast))
    fit_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    macro = induce_root_macro(observations)
    candidate = json.loads(json.dumps(asdict(macro))) if macro is not None else None
    ch = digest(candidate) if candidate is not None else None
    induction_ns = perf_counter_ns() - started

    rows = []
    scores = []
    started = perf_counter_ns()
    if macro is not None:
        # No validation verdict is consumed until every validation search is done.
        pending = []
        for phase_seed in manifest["seeds"]:
            for i, task in enumerate(validation_tasks):
                for arm in ARMS:
                    before = perf_counter_ns()
                    result = search(task, budget, phase_seed + i * 101, "evolution",
                                    macro=macro if arm == "macro" else None,
                                    known_composition=arm == "known")
                    elapsed = perf_counter_ns() - before
                    pending.append((phase_seed, task, arm, result, elapsed))
        for phase_seed, task, arm, result, elapsed in pending:
            before = perf_counter_ns()
            out = task.holdout_score(result.selected.candidate)
            score_ns = perf_counter_ns() - before
            rows.append({
                "seed": phase_seed, "task_id": task.task_id, "arm": arm,
                "manifest_sha256": mh, "candidate_sha256": ch,
                "comparator_sha256": manifest["comparators"][arm],
                "proposal_cap": budget, "attempt_cap": budget * 10,
                "exact_success": int(out.exact_fraction == 1),
                "proposals": result.proposals, "attempts": result.attempts,
                "macro_attempts": result.macro_attempts,
                "macro_proposals": result.macro_proposals, "search_ns": elapsed,
            })
            scores.append({
                "seed": phase_seed, "task_id": task.task_id, "arm": arm,
                "selected_ast": result.selected.candidate.serial(),
                "exact_fraction": out.exact_fraction, "score_ns": score_ns,
            })
    validation_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    decision = decide_utility(manifest, candidate, rows)
    gate_ns = perf_counter_ns() - started
    fit_totals = {key: sum(r[key] for r in training)
                  for key in ("proposals", "attempts", "search_ns")}
    validation_totals = {
        arm: {key: sum(r[key] for r in rows if r["arm"] == arm)
              for key in ("proposals", "attempts", "macro_attempts",
                          "macro_proposals", "search_ns")}
        for arm in ARMS
    }
    scoring_ns = sum(r["score_ns"] for r in scores)
    # Retain the complete report even if no macro is induced or the gate rejects.
    return {
        "development_seed": seed, "manifest": manifest,
        "source_hashes": sources, "candidate": candidate, "training": training,
        "surviving_training_observations": len(observations),
        "validation_rows": rows, "validation_scores": scores,
        "decision": decision, "validation_skipped_no_macro": macro is None,
        "fit_totals": fit_totals, "validation_work_totals": validation_totals,
        "cost_ns": {"setup": setup_ns, "fit": fit_ns, "induction": induction_ns,
                    "validation": validation_ns, "validation_scoring": scoring_ns,
                    "gate": gate_ns, "total_through_gate": perf_counter_ns() - start},
        "final_search_performed": False,
        "independent_provenance_authenticated": False,
        "matched_actual_compute_established": False,
        "milestone_claim_eligible": False,
    }


def run_dev2_validation(seed):
    """Fixed public DEV-2 configuration; does not execute final evaluation."""
    if type(seed) is not int or seed not in DEVELOPMENT_SEEDS:
        raise ValueError("seed not in frozen DEV-2")
    training, final = make_tasks(seed, FIT_TASKS + VALIDATION_TASKS, FINAL_TASKS)
    report = run_validation(training[:FIT_TASKS], training[FIT_TASKS:],
                            [t.task_id for t in final], seed=seed, budget=PROPOSAL_CAP)
    report["protocol_sha256"] = digest(protocol_descriptor())
    return report
