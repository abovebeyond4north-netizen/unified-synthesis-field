"""Fail-closed eligibility checks for empirical learning or efficiency claims.

This module does not score candidates or inspect concealed targets.  It only
audits already-produced, task-level rows and the preregistered execution
contract.  Passing this preflight is necessary, never sufficient, for a claim.
"""
from __future__ import annotations

from collections import Counter
import math
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
    paired_advantage_ci_lower_bound: float | None = None,
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
    required_rows = [row for row in materialized if row.get("mode") in REQUIRED_ARMS]

    def valid_outcome(value: object) -> bool:
        return (
            type(value) in (int, float)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )

    valid_task_ids = all(
        isinstance(row.get("task_id"), str) and bool(row["task_id"].strip())
        for row in required_rows
    )
    valid_outcomes = all(valid_outcome(row.get("holdout_exact")) for row in required_rows)
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
        if valid_outcome(row.get("holdout_exact")) and row.get("holdout_exact") == 1.0
    }
    retained_ids = {
        task_id for task_id in positive_ids
        if valid_outcome(candidate.get(task_id, {}).get("holdout_exact"))
        and candidate.get(task_id, {}).get("holdout_exact") == 1.0
    }
    lost_ids = sorted(positive_ids - retained_ids)

    paired_mean_difference = None
    if paired_tasks and valid_task_ids and valid_outcomes:
        paired_mean_difference = sum(
            float(candidate[task_id]["holdout_exact"])
            - float(baseline[task_id]["holdout_exact"])
            for task_id in sorted(tasks_by_arm["adaptive"])
        ) / expected_test_tasks

    valid_ci_lower_bound = valid_outcome(paired_advantage_ci_lower_bound)
    paired_advantage = paired_mean_difference is not None and paired_mean_difference > 0.0
    statistical_requirement = valid_ci_lower_bound and float(paired_advantage_ci_lower_bound) > 0.0

    checks = {
        "complete_required_arms": complete_arms,
        "valid_task_ids": valid_task_ids,
        "unique_task_ids_per_arm": unique_task_ids,
        "task_ids_paired_across_arms": paired_tasks,
        "valid_correctness_outcomes": valid_outcomes,
        "positive_no_memory_baseline_successes": len(positive_ids) > 0,
        "all_positive_baseline_successes_retained": not lost_ids,
        "configured_search_budget_per_task": configured_budget_per_task,
        "matched_wall_budget_enforced": matched_wall_budget_enforced is True,
        "concealed_evaluator": concealed_evaluator is True,
        "independent_administration": independent_administration is True,
    }
    blocking = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    claim_checks = {
        "preliminary_eligibility": not blocking,
        "positive_paired_candidate_advantage": paired_advantage,
        "bootstrap_95pct_lower_bound_exceeds_zero": statistical_requirement,
    }
    claim_blocking = [name for name, value in claim_checks.items() if not value]
    return {
        "eligible": not blocking,
        "blocking_reasons": blocking,
        "claim_accepted": not claim_blocking,
        "claim_blocking_reasons": claim_blocking,
        "required_arms": list(REQUIRED_ARMS),
        "no_memory_ablation": "cold_adaptive",
        "non_learning_comparator": "random",
        "expected_test_tasks": expected_test_tasks,
        "observed_rows_by_arm": {arm: counts[arm] for arm in REQUIRED_ARMS},
        "positive_baseline_successes": len(positive_ids),
        "retained_positive_baseline_successes": len(retained_ids),
        "lost_positive_baseline_task_ids": lost_ids,
        "paired_mean_difference": paired_mean_difference,
        "paired_advantage_ci_lower_bound": paired_advantage_ci_lower_bound,
        "checks": checks,
        "claim_checks": claim_checks,
        "interpretation": "preliminary eligibility and claim acceptance are necessary gates, not proof of independence, learning, or general intelligence",
    }
