"""Exploratory training-only macro reuse; not the frozen USF benchmark.

The original search policy, grammar and candidate scoring stay unchanged
unless a frozen macro is explicitly supplied to the evolutionary arm.
Public development tasks are not independent concealed evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from time import perf_counter_ns

from .core import make_tasks, sha256
from .macro_induction import induce_root_macro
from .search import search


def run_macro_trial(training_tasks, test_tasks, *, seed: int, budget: int):
    """Train once on disjoint task IDs, freeze, then compare two search arms.

    Task identity and phase ordering are checked, not independently authenticated.
    Equal candidate/attempt caps do not establish equal actual compute.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(budget) is not int or not 6 <= budget <= 2000:
        raise ValueError("budget must be an integer in [6, 2000]")
    train, test = tuple(training_tasks), tuple(test_tasks)
    if not 1 <= len(train) <= 32 or not 1 <= len(test) <= 32:
        raise ValueError("each split must contain 1..32 tasks")
    ids = [task.task_id for task in train + test]
    if any(type(i) is not str or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("training and test task IDs must be unique and disjoint")

    started = perf_counter_ns()
    observations = []
    training = []
    for i, task in enumerate(train):
        result = search(task, budget, seed + i * 101 + 7, "evolution", train=True)
        training.append({
            "task_id": task.task_id, "proposals": result.proposals,
            "attempts": result.attempts, "challenge_survived": result.selected.survived,
        })
        if result.selected.survived:
            observations.append((task.task_id, result.selected.candidate.serial()))
    training_ns = perf_counter_ns() - started

    started = perf_counter_ns()
    macro = induce_root_macro(observations)
    induction_ns = perf_counter_ns() - started
    frozen = asdict(macro) if macro is not None else None
    commitment = sha256(frozen)

    # Finish all search calls before any test holdout is scored.
    pending = []
    for i, task in enumerate(test):
        task_seed = seed + 100000 + i * 101
        for arm, candidate_macro in (("primitive", None), ("macro", macro)):
            started = perf_counter_ns()
            result = search(task, budget, task_seed, "evolution", macro=candidate_macro)
            search_ns = perf_counter_ns() - started
            pending.append((task, arm, result, search_ns))

    rows = []
    for task, arm, result, search_ns in pending:
        started = perf_counter_ns()
        heldout = task.holdout_score(result.selected.candidate)
        score_ns = perf_counter_ns() - started
        rows.append({
            "task_id": task.task_id, "arm": arm, "exact_success": int(heldout.exact_fraction == 1),
            "proposals": result.proposals, "attempts": result.attempts,
            "macro_attempts": result.macro_attempts, "macro_proposals": result.macro_proposals,
            "search_ns": search_ns, "holdout_scoring_ns": score_ns,
        })
    if sha256(asdict(macro) if macro is not None else None) != commitment:
        raise RuntimeError("macro changed after freeze")

    totals = {
        arm: {field: sum(row[field] for row in rows if row["arm"] == arm)
              for field in ("exact_success", "proposals", "attempts", "macro_attempts",
                            "macro_proposals", "search_ns", "holdout_scoring_ns")}
        for arm in ("primitive", "macro")
    }
    return {
        "status": "EXPLORATORY_MACRO_COMPARISON" if macro is not None else "NO_MACRO_INDUCED",
        "seed": seed, "candidate_cap_per_task": budget, "attempt_cap_per_task": budget * 10,
        "shared_training_ns": training_ns, "induction_ns": induction_ns,
        "training": training, "surviving_training_observations": len(observations),
        "frozen_macro": frozen, "macro_commitment": commitment,
        "test_rows": rows, "arm_totals": totals,
        "paired_exact_delta": (totals["macro"]["exact_success"] - totals["primitive"]["exact_success"]) / len(test),
        "independent_provenance_authenticated": False,
        "matched_actual_compute_established": False,
        "formal_verification_performed": False,
        "milestone_claim_eligible": False,
        "limitations": [
            "One-process public development experiment, not concealed independent evaluation",
            "Macro expansion is scored by the original primitive AST cost, not a new MDL encoding",
            "Matched configured caps are not matched measured compute; setup costs reported separately",
            "A macro-free run is retained as a negative result",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--train-tasks", type=int, default=20)
    parser.add_argument("--test-tasks", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.train_tasks <= 32 or not 1 <= args.test_tasks <= 32:
        parser.error("each split must contain 1..32 tasks")
    train, test = make_tasks(args.seed, args.train_tasks, args.test_tasks)
    print(json.dumps(run_macro_trial(train, test, seed=args.seed, budget=args.budget),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
