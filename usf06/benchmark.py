"""Equal-budget, seeded holdout benchmark with provenance and paired bootstrap CI."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import time
import uuid

from .core import canonical_json, make_tasks, sha256
from .ledger import Ledger
from .logos import LogosGate
from .pareto import FrontierItem, VerifiedParetoArchive
from .preflight import evaluation_claim_preflight, paired_bootstrap_interval
from .search import AdaptivePolicy, search


def run_benchmark(db_path: str, report_path: str, *, seed: int = 42, budget: int = 160,
                  n_train: int = 10, n_test: int = 6, mathlib_project: str | None = None):
    if budget < 6 or n_train < 1 or n_test < 2:
        raise ValueError("Invalid benchmark parameters")
    train, test = make_tasks(seed, n_train, n_test)
    run_id = uuid.uuid4().hex
    gate = LogosGate(mathlib_project)
    cfg = {"seed": seed, "budget_per_task": budget, "train_tasks": n_train,
           "test_tasks": n_test, "methods": ["random", "evolution", "cold_adaptive", "adaptive"],
           "allowed_actions": 8, "holdout_policy": "final_only_no_updates",
           "run_version": "USF-0.6.2", "python": platform.python_version(),
           "lean_version": gate.version}
    train_commitment = sha256([t.reveal() for t in train])
    holdout_commitment = sha256([t.reveal() for t in test])
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    source_commitment = sha256([{
        "module": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
    } for p in source_files])
    cfg["source_bundle_sha256"] = source_commitment
    policy = AdaptivePolicy()
    report_path = str(report_path)
    with Ledger(db_path) as ledger:
        ledger.append(run_id, "manifest", {
            "config": cfg, "train_task_commitment": train_commitment,
            "sealed_test_commitment": holdout_commitment,
            "protocol_id": "USF-V06-TOY-POLY-1", "code_release": "0.6.2",
            "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        training = []
        for i, task in enumerate(train):
            r = search(task, budget, seed=seed + i*101 + 7, mode="adaptive", policy=policy,
                       train=True, ledger=ledger, run_id=run_id)
            training.append({"task_id": task.task_id, "family": task.family,
                             "train_exact": r.selected.result.exact_fraction,
                             "challenge_survived": r.selected.survived,
                             "proposals": r.proposals, "attempts": r.attempts})
        frozen_policy = policy.snapshot()
        ledger.append(run_id, "policy_frozen", {"policy": frozen_policy, "before_heldout": True})

        rows = []
        formal_archive = VerifiedParetoArchive()
        for i, task in enumerate(test):
            for mode in ("random", "evolution", "cold_adaptive", "adaptive"):
                # Matched RNG seeds across methods, no learning or feedback from test scores.
                r = search(task, budget, seed=seed+10000+i*101,
                           mode="adaptive" if mode == "cold_adaptive" else mode,
                           policy=AdaptivePolicy() if mode == "cold_adaptive" else (policy if mode == "adaptive" else None),
                           train=False, ledger=ledger, run_id=run_id, label=mode)
                out = task.holdout_score(r.selected.candidate)
                proof_status = "NOT_SUBMITTED"
                axiom_audit_pass = False
                script = None
                formal_elapsed_s = 0.0
                if r.selected.survived and out.exact_fraction == 1.0:
                    theorem = f"usf_{mode}_{task.task_id.lower()}"
                    formal = gate.verify(r.selected.candidate, task._target, theorem)
                    proof_status = formal.status
                    axiom_audit_pass = formal.axiom_audit_pass
                    script = formal.lean_source
                    formal_elapsed_s = formal.elapsed_s
                    if formal.kernel_verified and formal.axiom_audit_pass:
                        formal_archive.admit(FrontierItem(r.selected.candidate_id,
                            r.selected.result.compression_gain_bits, out.exact_fraction,
                            r.attempts + r.proposals,
                            formal.kernel_verified, formal.axiom_audit_pass, r.selected.survived))
                row = {"task_id": task.task_id, "family": task.family, "mode": mode,
                       "candidate_id": r.selected.candidate_id,
                       "expression": str(r.selected.candidate),
                       "train_exact": r.selected.result.exact_fraction,
                       "falsification_survived": r.selected.survived,
                       "holdout_exact": out.exact_fraction,
                       "holdout_mae": out.mae,
                       "mdl_bits": r.selected.result.mdl_bits,
                       "compression_gain_bits": r.selected.result.compression_gain_bits,
                       "proposals": r.proposals, "attempts": r.attempts,
                       "formal_elapsed_s": round(formal_elapsed_s, 6),
                       "lean_status": proof_status,
                       "axiom_audit_pass": axiom_audit_pass}
                rows.append(row)
                ledger.append(run_id, "sealed_evaluation", {**row, "proof_script": script},
                              r.selected.candidate_id)
        # Reveal sealed tasks only AFTER all modes and policies have finished.
        sealed_reveal = {"training": [t.reveal() for t in train],
                         "test": [t.reveal() for t in test]}
        assert sha256(sealed_reveal["training"]) == train_commitment
        assert sha256(sealed_reveal["test"]) == holdout_commitment
        ledger.append(run_id, "reveal_after_evaluation", sealed_reveal)

        def mean(field, mode, family=None):
            vs = [r[field] for r in rows if r["mode"] == mode and (family is None or r["family"] == family)]
            return round(sum(vs)/len(vs), 4) if vs else None

        methods = {mode: {"holdout_exact_mean": mean("holdout_exact", mode),
                          "holdout_mae_mean": mean("holdout_mae", mode),
                          "quadratic_exact": mean("holdout_exact", mode, "quadratic"),
                          "unseen_cubic_exact": mean("holdout_exact", mode, "cubic"),
                          "verified_count": sum(r["lean_status"] == "KERNEL_VERIFIED" for r in rows if r["mode"] == mode)}
                   for mode in ("random", "evolution", "cold_adaptive", "adaptive")}
        adapted = {r["task_id"]: r for r in rows if r["mode"] == "adaptive"}
        baseline = {r["task_id"]: r for r in rows if r["mode"] == "cold_adaptive"}
        paired = [adapted[t.task_id]["holdout_exact"] - baseline[t.task_id]["holdout_exact"] for t in test]
        delta = sum(paired)/len(paired)
        bootstrap_seed = seed + 999
        lower, upper = paired_bootstrap_interval(paired, bootstrap_seed)
        claim_preflight = evaluation_claim_preflight(
            rows,
            expected_test_tasks=len(test),
            configured_budget_per_task=budget,
            # The in-process v0.6.2 harness cannot establish these properties.
            concealed_evaluator=False,
            independent_administration=False,
            matched_wall_budget_enforced=False,
            bootstrap_seed=bootstrap_seed,
        )
        report = {"run_id": run_id, "config": cfg,
                  "train_task_commitment": train_commitment,
                  "sealed_test_commitment": holdout_commitment,
                  "training": training, "frozen_policy": frozen_policy,
                  "methods": methods,
                  "claim_preflight": claim_preflight,
                  "comparison_trained_vs_cold_adaptive": {
                      "paired_mean_difference": round(delta, 4),
                      "bootstrap_95pct_CI": [lower, upper],
                      "n_test_tasks": len(test),
                      "interpretation": "toy benchmark only; no evidence of general intelligence"},
                  "verified_pareto_archive": [asdict(v) for v in formal_archive.items],
                  "test_rows": rows,
                  "seal_revealed_only_after_evaluation": True,
                  "limitations": ["toy integer polynomial tasks only",
                      "learner uses handwritten mutations, not GATv2/MCTS",
                      "holdout secrecy enforced by program control flow, not process isolation",
                      "test task family coefficients revealed after evaluation for reproducibility",
                      "hash chain is not externally signed or anchored",
                      "a formal proof requires native Lean 4 and Mathlib",
                      "Pareto compute cost counts search proposals plus mutation attempts, not physical FLOPs"]}
        ledger.append(run_id, "benchmark_report", {"sha256": sha256(report),
                                                     "summary": report["methods"]})
        verified, count = ledger.verify_integrity()
        report["ledger_integrity"] = {"verified": verified, "events": count}
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
