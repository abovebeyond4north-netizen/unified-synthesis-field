# Unified Synthesis Field (USF) — v0.6.2

Public research repository for reproducible neuro-symbolic discovery experiments. CI: `.github/workflows/python.yml` and `.github/workflows/lean.yml`. Roadmap: `docs/ROADMAP.md`. The Lean job is an **initial smoke test**, not proof of autonomous discovery.

## Status: No demonstrated self-improvement

The five-seed trained-versus-cold comparison had **0 percentage-point mean improvement**; native Lean verification was not available in the original local runs. GitHub Actions results must be inspected separately.

## Experimental infrastructure

This is a **runnable research harness**, not a demonstration of artificial general intelligence, novel theorem discovery, GATv2, or MCTS. It uses **Python 3.10+ standard library only**; native Lean 4 and Mathlib are optional for kernel-checked polynomial equivalence proofs. When Lean cannot run, formal admission is **disabled**.

## Delivered modules

| Module | Purpose |
|---|---|
| `usf06/core.py` | Typed integer ASTs, restricted arithmetic, task oracle, specified two-part integer MDL code, candidate-specific empirical counterexample generation |
| `usf06/ledger.py` | Append-only SQLite event log, timestamp-inclusive SHA-256 hash chain, immutability triggers, parent and duplicate validation, SQL discovery view |
| `usf06/logos.py` | Generated Lean polynomial equivalence theorem (`ring`), native `lake env lean`, trusted axiom allowlist via `#print axioms`, fail-closed status |
| `usf06/pareto.py` | Nondominated archive that **refuses** non-kernel-verified, unaudited, or falsified hypotheses |
| `usf06/search.py` | Equal-budget random/evolutionary/weighted-operator search, frozen learned policy during testing |
| `usf06/benchmark.py` | Generated training and sealed test tasks, policy freeze, matched before/after comparison, paired bootstrap summary, post-evaluation disclosure of holdout task definitions |
| `tests/test_usf06.py` | 18 local tests, including overfitting counterexample, hash tampering, no holdout policy update, axiom audit, and benchmark disclosure order |

## Quick start

From this folder:

```bash
python -m unittest discover -s tests -v
python -m usf06 benchmark --db your_ledger.sqlite --report your_report.json --seed 9006 --budget 160 --train-tasks 10 --test-tasks 6
python -m usf06 ledger-check --db your_ledger.sqlite
```

Run with native Mathlib:

```bash
# In a separate location, install Lean 4 using elan, then create a Mathlib Lake project.
lake new usf_math math
cd usf_math
lake exe cache get
cd ..
# From the USF v0.6 package folder:
python -m usf06 benchmark --db native_ledger.sqlite --report native_report.json --mathlib-project /absolute/path/to/usf_math
```

This installation path has **not** been executed in this environment. Native Lean 4 was unavailable; all formal outcomes in the included runs are unverified. The generated statements prove equivalence between discovered integer-polynomial expressions and evaluator-held target expressions, **not** Euler's identity and not newly invented mathematics.

## Benchmark contract

- Training tasks: 10 affine/quadratic function-synthesis problems sampled from a deterministic seed.
- Test tasks: 6 unseen problems; half quadratic, half cubic (cubic family not used in training).
- Search: at most **160 unique candidate evaluations per task** plus bounded mutation attempts. Duplicate attempts count toward compute telemetry but not the candidate-evaluation budget.
- Four comparators: uniform random mutation, elite-selection evolution, **cold** adaptive architecture with untrained action weights, and **trained** adaptive architecture with weights learned on training tasks.
- Sealed oracle: the learner may query training observations and a separate adversarial-probe pool, but **never the final holdout evaluation**. Test-policy learning is disabled. The code does not provide cryptographic or process-level separation from an untrusted learner.
- Test data are committed by hash at run start and revealed as ledger events **after** all evaluations. Reproducibility disclosures make subsequent reuse of these same test seeds unsuitable as a blind confirmation test.

## Precise mathematical definitions

The code uses a fixed *two-part prefix-code surrogate* for MDL, not Kolmogorov complexity:

- Each AST opcode uses three bits; integer constants use an Elias-gamma length of zig-zag encoded signed integer plus one.
- Training observations have known inputs `x` and integer outputs `y`.
- Residuals `y - h(x)` use the same signed-integer prefix length.
- `L_total = L(h) + sum_i L(y_i - h(x_i))`.
- `compression_gain = sum_i L(y_i) - L_total` relative to a fixed zero-predictor baseline.

This is *not* conditional MDL `L(h | K_t)` unless the background-knowledge coding prior is explicitly supplied and accounted for. No candidate is admitted based solely on a numerical test.

For `p` to dominate `q` in the Pareto archive, all three objectives must be weakly better and at least one strictly better: `(compression_gain ↑, transfer ↑, computational_work ↓)`.

## Ledger and threat model

Every append records: run ID, event kind, candidate/parent IDs, canonical JSON payload, previous hash, UTC timestamp, and SHA-256 digest. `UPDATE`/`DELETE` triggers protect the normal SQLite interface. The `ledger-check` command validates the entire chain. Candidate history and subsequent sealed evaluation events are distinct, avoiding post-hoc changes to historical candidate values.

**Limit:** A privileged adversary can drop SQLite triggers and rewrite/re-hash the entire log. For strong tamper evidence, periodically anchor signed root hashes in an independent store. No signing or external anchoring is implemented. The task oracle and search run in one Python process: the test boundary is an experimental software contract, not a hardened security boundary.

## Observed results

See `EXPERIMENT_REPORT.md`, `results_v062.json`, and `replications/summary.json`. The learned controller **did not demonstrate improvement** over its cold equivalent on five independent task seeds. No native Lean theorem was certified.

## Next smallest defensible work items

1. Add a *truly blinded* evaluation service in a separate process and signed ledger checkpoints.
2. Install/pin Mathlib and verify the generated integer polynomial proofs with native Lean.
3. Improve action-credit assignment on a **development-only split** and retain an entirely new hidden task suite for unbiased confirmation.
4. Introduce MCTS/GATv2 only after simpler control algorithms establish measurable benefit; expand beyond hand-authored operators to learned reusable abstractions.
