"""Read-only, fail-closed resource audit of published USF benchmark reports.

The receipt describes *reported* work; it does not authenticate provenance,
prove actual FLOPs, verify Lean proofs, or qualify a milestone claim.
Run: python tools/audit_resource_receipts.py results_v062.json replications/*.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

ARMS = ("random", "evolution", "cold_adaptive", "adaptive")
INITIAL_PROPOSALS = 6


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}")
    return value


def _binary_exact(value: Any, label: str) -> int:
    if type(value) is int and value in (0, 1):
        return value
    if type(value) is float and math.isfinite(value) and value in (0.0, 1.0):
        return int(value)
    raise ValueError(f"{label}: expected binary numeric value 0 or 1")


def _duration(value: Any, label: str) -> float:
    # Reject absurd or overflowing int evidence before any float coercion.
    if type(value) is int:
        if 0 <= value <= 10**12:
            return float(value)
    elif type(value) is float and math.isfinite(value) and 0.0 <= value <= 1e12:
        return value
    raise ValueError(f"{label}: expected finite duration in [0, 1e12]")


def audit_report(report: dict[str, Any]) -> dict[str, Any]:
    """Require exhaustive four-arm pairing and report, never infer, resources."""
    if not isinstance(report, dict) or not isinstance(report.get("config"), dict):
        raise ValueError("missing config")
    config = report["config"]
    seed = _integer(config.get("seed"), "config.seed")
    budget = _integer(config.get("budget_per_task"), "config.budget_per_task", 1)
    tasks = _integer(config.get("test_tasks"), "config.test_tasks", 1)
    rows = report.get("test_rows")
    if type(rows) is not list or len(rows) != tasks * len(ARMS):
        raise ValueError("unexpected number of test rows")
    by_arm: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    family_by_task: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index}: not an object")
        mode, task_id, family = row.get("mode"), row.get("task_id"), row.get("family")
        if mode not in ARMS:
            raise ValueError(f"row {index}: unexpected arm {mode!r}")
        if type(task_id) is not str or not task_id.strip():
            raise ValueError(f"row {index}: invalid task_id")
        if type(family) is not str or not family.strip():
            raise ValueError(f"row {index}: invalid family")
        if task_id in by_arm[mode]:
            raise ValueError(f"row {index}: duplicate ({mode}, {task_id})")
        if task_id in family_by_task and family_by_task[task_id] != family:
            raise ValueError(f"row {index}: family mismatch for task {task_id}")
        family_by_task[task_id] = family
        proposals = _integer(row.get("proposals"), f"row {index}.proposals", 1)
        if proposals > budget:
            raise ValueError(f"row {index}: proposals exceed configured budget")
        attempts = _integer(row.get("attempts"), f"row {index}.attempts")
        if attempts < max(0, proposals - INITIAL_PROPOSALS):
            raise ValueError(
                f"row {index}: attempts cannot produce reported proposals"
            )
        _binary_exact(row.get("holdout_exact"), f"row {index}.holdout_exact")
        _duration(row.get("formal_elapsed_s"), f"row {index}.formal_elapsed_s")
        if "setup_inclusive_wall_s" in row:
            _duration(row["setup_inclusive_wall_s"], f"row {index}.setup_inclusive_wall_s")
        by_arm[mode][task_id] = row
    expected = set(family_by_task)
    if len(expected) != tasks or any(set(by_arm[arm]) != expected for arm in ARMS):
        raise ValueError("unpaired or missing test task identities")
    has_wall = all("setup_inclusive_wall_s" in r for r in rows)
    arms = {}
    for mode in ARMS:
        entries = list(by_arm[mode].values())
        arms[mode] = {
            "reported_proposals": sum(r["proposals"] for r in entries),
            "reported_mutation_attempts": sum(r["attempts"] for r in entries),
            "reported_formal_seconds_only": round(sum(r["formal_elapsed_s"] for r in entries), 6),
            "exact_task_successes": sum(r["holdout_exact"] == 1 for r in entries),
            "reported_setup_inclusive_wall_seconds": (
                round(sum(r["setup_inclusive_wall_s"] for r in entries), 6)
                if has_wall else None
            ),
        }
    paired_sum = sum(
        by_arm["adaptive"][task_id]["holdout_exact"]
        - by_arm["cold_adaptive"][task_id]["holdout_exact"]
        for task_id in expected
    )
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"adaptive": 0, "cold_adaptive": 0, "tasks": 0})
    for task_id, family in family_by_task.items():
        by_family[family]["tasks"] += 1
        for mode in ("adaptive", "cold_adaptive"):
            by_family[family][mode] += int(by_arm[mode][task_id]["holdout_exact"] == 1)
    return {
        "seed": seed,
        "task_count": tasks,
        "budget_per_task_configured": budget,
        "arm_order": list(ARMS),
        "arms": arms,
        "paired_trained_minus_cold_mean": round(paired_sum / tasks, 6),
        "exact_successes_by_family": dict(sorted(by_family.items())),
        "reported_setup_inclusive_wall_complete": has_wall,
        "actual_compute_equivalence_established": False,
        "independent_provenance_authenticated": False,
        "milestone_claim_eligible": False,
        "limits": [
            "Self-reported benchmark data; no independently authenticated source or run commitment",
            "Proposal equality does not imply equality of mutation attempts, FLOPs, or wall time",
            "Formal elapsed seconds exclude search, training, setup, and possibly compilation",
            "Observed wall time, if present, cannot by itself establish enforced matched wall budgets",
            "Recorded Lean status is not independently kernel-verified by this auditor",
        ] + ([] if has_wall else ["No setup-inclusive wall-time field in every task/arm row"]),
    }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def audit_files(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one report required")
    receipts = []
    seen_seeds: set[int] = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
        receipt = audit_report(report)
        if receipt["seed"] in seen_seeds:
            raise ValueError(f"duplicate seed {receipt['seed']}")
        seen_seeds.add(receipt["seed"])
        receipts.append(receipt)
    total_tasks = sum(r["task_count"] for r in receipts)
    return {
        "status": "STRUCTURALLY_VALID_REPORTED_DATA_ONLY",
        "receipts": sorted(receipts, key=lambda r: r["seed"]),
        "aggregate": {
            "runs": len(receipts),
            "paired_tasks": total_tasks,
            "arm_totals": {
                mode: {
                    key: sum(r["arms"][mode][key] for r in receipts)
                    for key in ("reported_proposals", "reported_mutation_attempts", "exact_task_successes")
                }
                for mode in ARMS
            },
            "paired_trained_minus_cold_mean": round(
                sum(
                    r["arms"]["adaptive"]["exact_task_successes"]
                    - r["arms"]["cold_adaptive"]["exact_task_successes"]
                    for r in receipts
                )
                / total_tasks, 6
            ),
            "milestones_a_b_c_demonstrated": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_files(args.reports)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "REJECTED", "reason": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

