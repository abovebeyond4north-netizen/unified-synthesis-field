"""CLI: python -m usf06 benchmark / ledger-check"""
import argparse
import json
from .benchmark import run_benchmark
from .ledger import Ledger


def main():
    parser = argparse.ArgumentParser(description="USF v0.6 fail-closed research harness")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark", help="Run the seeded toy discovery benchmark")
    bench.add_argument("--db", default="usf_ledger.sqlite")
    bench.add_argument("--report", default="usf_report.json")
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--budget", type=int, default=160)
    bench.add_argument("--train-tasks", type=int, default=10)
    bench.add_argument("--test-tasks", type=int, default=6)
    bench.add_argument("--mathlib-project", default=None)
    check = sub.add_parser("ledger-check", help="Audit the append-only hash chain")
    check.add_argument("--db", required=True)
    args = parser.parse_args()
    if args.command == "benchmark":
        r = run_benchmark(args.db, args.report, seed=args.seed, budget=args.budget,
                          n_train=args.train_tasks, n_test=args.test_tasks,
                          mathlib_project=args.mathlib_project)
        print(json.dumps({"run_id": r["run_id"], "methods": r["methods"],
                          "paired": r["comparison_trained_vs_cold_adaptive"],
                          "integrity": r["ledger_integrity"]}, indent=2))
    else:
        with Ledger(args.db) as ledger:
            integrity, count = ledger.verify_integrity()
        print(json.dumps({"ok": integrity, "events_or_failing_sequence": count}))
        if not integrity:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
