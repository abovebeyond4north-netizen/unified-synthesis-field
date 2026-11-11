"""Fail-closed eligibility checks for empirical learning or efficiency claims.

This module does not score candidates or inspect concealed targets.  It only
audits already-produced, task-level rows and the preregistered execution
contract.  Passing this preflight is necessary, never sufficient, for a claim.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


REQUIRED_ARMS = ("random", "cold_adaptive", "adaptive")


def evaluation_claim_preflight(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_test_tasks: int,
    configured_budget_per_task: int,
    concealed_evaluator: bool = False,
    independent_administration: bool = False,
    matched_wall_budget_enforced: bool = False,
) -> dict[str, object]:
    """Return a canonical, fail-closed claim-eligibility summary.

    ``cold_adaptive`` is the causal no-memory ablation: it uses the same
    adaptive search implementation with a fresh policy.  Random is retained as
    a non-learning comparator.  The function deliberately cannot infer
    concealment, independence, or wall-budget enforcement from result rows.
    Those properties must be supplied by the frozen execution contract.
    """
    if expected_test_tasks < 1 or configured_budget_per_task < 1:
        raise ValueError("expected tasks and configured budget must be positive")

    materialized = [dict(row) for row in rows]
    counts = Counter(str(row.get("mode")) for row in materialized)
    tasks_by_arm = {
        arm: {str(row.get("task_id")) for row in materialized if row.get("mode") == arm}
        for arm in REQUIRED_ARMS
    }
    complete_arms = all(counts[arm] == expected_test_tasks for arm in REQUIRED_ARMS)
    unique_task_ids = all(len(tasks_by_arm[arm]) == expected_test_tasks for arm in REQUIRED_ARMS)
    paired_tasks = (
        complete_arms
        and unique_task_ids
        and len({frozenset(tasks_by_arm[arm]) for arm in REQUIRED_ARMS}) == 1
    )

    baseline = {
        str(row.get("task_id")): row
        for row in materialized
        if row.get("mode") == "cold_adaptive"
    }
    candidate = {
        str(row.get("task_id")): row
        for row in materialized
        if row.get("mode") == "adaptive"
    }
    positive_ids = {
        task_id for task_id, row in baseline.items()
        if row.get("holdout_exact") == 1.0
    }
    retained_ids = {
        task_id for task_id in positive_ids
        if candidate.get(task_id, {}).get("holdout_exact") == 1.0
    }
    lost_ids = sorted(positive_ids - retained_ids)

    checks = {
        "complete_required_arms": complete_arms,
        "unique_task_ids_per_arm": unique_task_ids,
        "task_ids_paired_across_arms": paired_tasks,
        "positive_no_memory_baseline_successes": len(positive_ids) > 0,
        "all_positive_baseline_successes_retained": not lost_ids,
        "configured_search_budget_per_task": configured_budget_per_task,
        "matched_wall_budget_enforced": bool(matched_wall_budget_enforced),
        "concealed_evaluator": bool(concealed_evaluator),
        "independent_administration": bool(independent_administration),
    }
    blocking = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    return {
        "eligible": not blocking,
        "blocking_reasons": blocking,
        "required_arms": list(REQUIRED_ARMS),
        "no_memory_ablation": "cold_adaptive",
        "non_learning_comparator": "random",
        "expected_test_tasks": expected_test_tasks,
        "observed_rows_by_arm": {arm: counts[arm] for arm in REQUIRED_ARMS},
        "positive_baseline_successes": len(positive_ids),
        "retained_positive_baseline_successes": len(retained_ids),
        "lost_positive_baseline_task_ids": lost_ids,
        "checks": checks,
        "interpretation": "necessary preflight only; not evidence of learning or general intelligence",
    }
