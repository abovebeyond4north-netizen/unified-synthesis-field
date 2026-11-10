# USF v0.6.2 — observed experimental outcomes

All evidence below comes from actual local Python executions. Native Lean 4 is unavailable in the execution environment; **zero kernel-verified discoveries** were recorded.

## Test verification

- `python -m unittest discover -s tests -q`: **18/18 passing**.
- Main benchmark: seed **42**, 10 training tasks, 6 test tasks, max 160 unique candidate evaluations per task; **5,468** hash-chain events, integrity verified.
- Five independent task seeds **9001–9005**, same budget/task protocol; **5,468 events per seed**, with separate SQLite ledgers and successful integrity validation.

## Main run — seed 42

| Search method | Mean held-out exact fraction | Held-out cubic exact fraction | Native Lean verified |
|---|---:|---:|---:|
| Random | 0.5000 | 0.0000 | 0 |
| Evolution | 0.3333 | 0.0000 | 0 |
| Cold adaptive | 0.5000 | 0.3333 | 0 |
| Trained adaptive | 0.3333 | 0.0000 | 0 |

Paired trained minus cold adaptive: **−0.1667** (bootstrap 95% interval **[−0.5000, 0.0000]** across six tasks). This is not evidence for improvement.

## Additional-seed replications

| Task seed | Trained exact | Cold exact | Paired difference | Native Lean verified |
|---|---:|---:|---:|---:|
| 9001 | 0.3333 | 0.3333 | +0.0000 | 0 |
| 9002 | 0.3333 | 0.1667 | +0.1667 | 0 |
| 9003 | 0.3333 | 0.5000 | −0.1667 | 0 |
| 9004 | 0.3333 | 0.3333 | +0.0000 | 0 |
| 9005 | 0.3333 | 0.3333 | +0.0000 | 0 |

**Across-seed mean difference: 0.0000**, with one positive, one negative, and three tied seeds. No positive longitudinal gain has been established. The set is small and the task family synthetic, so this does not establish a general absence of possible learning either.

## Interpretation

The current prototype demonstrates that its **testing and audit infrastructure can execute**, not that a strong autonomous theorem-discovery algorithm has emerged. The adaptive operator-learning strategy was not measurably more effective than an identical cold controller on these seeds. The cubic transfer task remained largely unsolved and, in the independent-seed replications, trained cubic success was zero.

The inability to certify kernel proofs is an infrastructure limitation, not a claim that any mathematical proposition has been proven false. Formal and empirical status are deliberately independent.

## Evidence limitations

- Small sample (6 held-out tasks per seed), synthetic integer-polynomial domain.
- Both learner and oracle are implemented in the same process; holdout hiding is enforced through code paths, not cryptographic access control.
- A specific mutation vocabulary and polynomial domain are manually designed; this is **not** representation invention from an unknown primitive substrate.
- Native Lean + Mathlib were not installed, so the proof generator and actual axiom audit remain **untested on a Lean kernel**.
- No GATv2, MCTS, open-ended autonomous curriculum, or broadly general discovery is part of this v0.6.2 runnable harness.
