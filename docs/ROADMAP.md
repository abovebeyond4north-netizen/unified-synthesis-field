# Milestones and acceptance gates

The goal is **empirically validated gains in verified discovery**, not proof of general intelligence from a known theorem.

## A — Verified rediscovery
- [ ] Pin and execute a native Mathlib verification test in GitHub Actions.
- [ ] Run candidate-specific proofs through `LogosGate` with `#print axioms` audit.
- [ ] Demonstrate blind, target-free rediscovery under a fixed search budget.

## B — Autonomous representation invention
- [ ] Search over representations not merely preselected target transformations.
- [ ] Record a reusable abstraction with independent proofs and provenance.
- [ ] Demonstrate improvement on several tasks relative to the available DSL baseline.

## C — Cross-domain transfer and learning
- [ ] Provide genuinely withheld task families in a separately controlled evaluator.
- [ ] Compare trained versus cold copies of the same controller with matched seeds and compute.
- [ ] Require independent replication, error bars, counterexample search, and signed audit checkpoints.

## Present limitations
- This repository contains a synthetic integer-polynomial search harness, **not** GATv2 or MCTS.
- The five additional v0.6.2 seeds show **0.00 average trained-minus-cold accuracy gain**.
- Local native Lean verification has **not** been performed; hosted CI status must be checked separately.
- Published task seeds and results must not be reused as secret final-confirmation tests.
