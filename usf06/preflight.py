"""Fail-closed eligibility checks for empirical learning or efficiency claims.

This module does not score candidates or inspect concealed targets.  It only
audits already-produced, task-level rows and the preregistered execution
contract.  Passing this preflight is necessary, never sufficient, for a claim.
"""
from __future__ import annotations

from collections import Counter
import hmac
import math
import random
from typing import Iterable, Mapping


REQUIRED_ARMS = ("random", "evolution", "cold_adaptive", "adaptive")
REQUIRED_PROVENANCE_RECEIPTS = (
    "protocol_sha256",
    "source_bundle_sha256",
    "target_commitment",
    "evaluator_sha256",
)
REQUIRED_CANDIDATE_EVALUATIONS_PER_TASK = 160
REQUIRED_BOOTSTRAP_ITERATIONS = 2000


def paired_bootstrap_interval(
    differences: list[float], seed: int, iterations: int = REQUIRED_BOOTSTRAP_ITERATIONS
) -> list[float | None]:
    """Recompute the deterministic paired-bootstrap interval used by the benchmark."""
    if not differences:
        return [None, None]
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("bootstrap iterations must be a positive integer")
    rng = random.Random(seed)
    n = len(differences)
    values = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    return [
        round(values[int(.025 * (iterations - 1))], 4),
        round(values[int(.975 * (iterations - 1))], 4),
    ]


def evaluation_claim_preflight(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_test_tasks: int,
    configured_budget_per_task: int,
    expected_task_order: Iterable[str] | None = None,
    concealed_evaluator: bool = False,
    independent_administration: bool = False,
    matched_wall_budget_enforced: bool = False,
    bootstrap_seed: int | None = None,
    bootstrap_iterations: int = REQUIRED_BOOTSTRAP_ITERATIONS,
    paired_advantage_ci_lower_bound: float | None = None,
    provenance_receipts: Mapping[str, object] | None = None,
    expected_provenance_receipts: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a canonical, fail-closed claim-eligibility summary.

    ``cold_adaptive`` is the causal no-memory ablation: it uses the same
    adaptive search implementation with a fresh policy.  Random and evolution
    are retained as distinct non-learning comparators.  The function cannot infer
    concealment, independence, wall-budget enforcement, the intended task
    order, or provenance authenticity from result rows.  Those properties must
    be supplied by the frozen execution contract.  Statistical acceptance is
    reported separately from complete Milestone A acceptance.
    """
    if type(expected_test_tasks) is not int or expected_test_tasks < 1:
        raise ValueError("expected tasks must be a positive integer")

    configured_budget_matches_frozen_contract = (
        type(configured_budget_per_task) is int
        and configured_budget_per_task == REQUIRED_CANDIDATE_EVALUATIONS_PER_TASK
    )

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
    observed_task_order_by_arm = {
        arm: [str(row.get("task_id")) for row in materialized if row.get("mode") == arm]
        for arm in REQUIRED_ARMS
    }
    task_order_receipt = None
    if expected_task_order is not None and not isinstance(expected_task_order, str):
        try:
            task_order_receipt = list(expected_task_order)
        except TypeError:
            pass
    valid_task_order_receipt = (
        task_order_receipt is not None
        and len(task_order_receipt) == expected_test_tasks
        and all(isinstance(task_id, str) and bool(task_id.strip()) for task_id in task_order_receipt)
        and len(set(task_order_receipt)) == expected_test_tasks
    )
    complete_arms = all(counts[arm] == expected_test_tasks for arm in REQUIRED_ARMS)
    unique_task_ids = all(len(tasks_by_arm[arm]) == expected_test_tasks for arm in REQUIRED_ARMS)
    paired_tasks = (
        complete_arms
        and unique_task_ids
        and len({frozenset(tasks_by_arm[arm]) for arm in REQUIRED_ARMS}) == 1
    )
    ordered_tasks = (
        valid_task_order_receipt
        and all(observed_task_order_by_arm[arm] == task_order_receipt for arm in REQUIRED_ARMS)
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

    paired_differences: list[float] = []
    paired_mean_difference = None
    if paired_tasks and ordered_tasks and valid_task_ids and valid_outcomes:
        paired_task_ids = [
            str(row["task_id"])
            for row in materialized
            if row.get("mode") == "adaptive"
        ]
        paired_differences = [
            float(candidate[task_id]["holdout_exact"])
            - float(baseline[task_id]["holdout_exact"])
            for task_id in paired_task_ids
        ]
        paired_mean_difference = sum(paired_differences) / expected_test_tasks

    valid_bootstrap_configuration = (
        type(bootstrap_seed) is int
        and type(bootstrap_iterations) is int
        and bootstrap_iterations == REQUIRED_BOOTSTRAP_ITERATIONS
    )
    bootstrap_interval: list[float | None] = [None, None]
    if paired_differences and valid_bootstrap_configuration:
        bootstrap_interval = paired_bootstrap_interval(
            paired_differences,
            bootstrap_seed,
            bootstrap_iterations,
        )
    computed_ci_lower_bound = bootstrap_interval[0]

    def valid_reported_bound(value: object) -> bool:
        return (
            type(value) in (int, float)
            and math.isfinite(float(value))
            and -1.0 <= float(value) <= 1.0
        )

    reported_bound_consistent = (
        paired_advantage_ci_lower_bound is None
        or (
            valid_reported_bound(paired_advantage_ci_lower_bound)
            and computed_ci_lower_bound is not None
            and math.isclose(
                float(paired_advantage_ci_lower_bound),
                computed_ci_lower_bound,
                abs_tol=1e-12,
            )
        )
    )
    paired_advantage = paired_mean_difference is not None and paired_mean_difference > 0.0
    statistical_requirement = (
        computed_ci_lower_bound is not None
        and computed_ci_lower_bound > 0.0
        and reported_bound_consistent
    )

    checks = {
        "complete_required_arms": complete_arms,
        "valid_task_ids": valid_task_ids,
        "unique_task_ids_per_arm": unique_task_ids,
        "task_ids_paired_across_arms": paired_tasks,
        "task_order_matches_frozen_receipt": ordered_tasks,
        "valid_correctness_outcomes": valid_outcomes,
        "positive_no_memory_baseline_successes": len(positive_ids) > 0,
        "all_positive_baseline_successes_retained": not lost_ids,
        "configured_search_budget_matches_frozen_contract": (
            configured_budget_matches_frozen_contract
        ),
        "matched_wall_budget_enforced": matched_wall_budget_enforced is True,
        "concealed_evaluator": concealed_evaluator is True,
        "independent_administration": independent_administration is True,
    }
    blocking = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    claim_checks = {
        "preliminary_eligibility": not blocking,
        "positive_paired_candidate_advantage": paired_advantage,
        "recorded_bootstrap_configuration": valid_bootstrap_configuration,
        "reported_bootstrap_lower_bound_consistent": reported_bound_consistent,
        "bootstrap_95pct_lower_bound_exceeds_zero": statistical_requirement,
    }
    claim_blocking = [name for name, value in claim_checks.items() if not value]

    recorded_receipts = (
        dict(provenance_receipts)
        if isinstance(provenance_receipts, Mapping)
        else {}
    )
    frozen_receipts = (
        dict(expected_provenance_receipts)
        if isinstance(expected_provenance_receipts, Mapping)
        else {}
    )

    def valid_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    provenance_checks = {
        name: (
            valid_sha256(recorded_receipts.get(name))
            and valid_sha256(frozen_receipts.get(name))
            and hmac.compare_digest(recorded_receipts[name], frozen_receipts[name])
        )
        for name in REQUIRED_PROVENANCE_RECEIPTS
    }
    receipt_strings_match = all(provenance_checks.values())
    milestone_a_checks = {
        "statistical_claim_accepted": not claim_blocking,
        "receipt_strings_match": receipt_strings_match,
        # This in-process preflight only compares caller-supplied strings.  It
        # cannot authenticate artifact bytes, commitment timing, evaluator
        # control, or an independently anchored attestation.
        "independently_authenticated_provenance": False,
    }
    milestone_a_blocking = [
        name for name, value in milestone_a_checks.items() if not value
    ]
    return {
        "eligible": not blocking,
        "blocking_reasons": blocking,
        "claim_accepted": not claim_blocking,
        "claim_blocking_reasons": claim_blocking,
        "milestone_a_accepted": not milestone_a_blocking,
        "milestone_a_blocking_reasons": milestone_a_blocking,
        "required_provenance_receipts": list(REQUIRED_PROVENANCE_RECEIPTS),
        "provenance_checks": provenance_checks,
        "receipt_strings_match": receipt_strings_match,
        "milestone_a_checks": milestone_a_checks,
        "required_arms": list(REQUIRED_ARMS),
        "no_memory_ablation": "cold_adaptive",
        "non_learning_comparator": "random",
        "evolutionary_comparator": "evolution",
        "expected_test_tasks": expected_test_tasks,
        "configured_budget_per_task": configured_budget_per_task,
        "required_candidate_evaluations_per_task": (
            REQUIRED_CANDIDATE_EVALUATIONS_PER_TASK
        ),
        "observed_rows_by_arm": {arm: counts[arm] for arm in REQUIRED_ARMS},
        "expected_task_order": task_order_receipt,
        "observed_task_order_by_arm": observed_task_order_by_arm,
        "positive_baseline_successes": len(positive_ids),
        "retained_positive_baseline_successes": len(retained_ids),
        "lost_positive_baseline_task_ids": lost_ids,
        "paired_mean_difference": paired_mean_difference,
        "paired_advantage_bootstrap_95pct_CI": bootstrap_interval,
        "paired_advantage_ci_lower_bound": computed_ci_lower_bound,
        "reported_paired_advantage_ci_lower_bound": paired_advantage_ci_lower_bound,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_iterations": bootstrap_iterations,
        "checks": checks,
        "claim_checks": claim_checks,
        "interpretation": "statistical claim acceptance and matching caller-supplied receipt strings do not authenticate provenance; Milestone A requires a separately controlled verifier that rehashes immutable artifacts and validates pre-run commitments and evaluator attestations",
    }
