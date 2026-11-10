# Contributing

Contributions should use a pull request, include focused tests, and retain counterexamples and negative results. Avoid changing existing held-out benchmark cases in response to results. Declare all theorem/axiom dependencies and label symbolic, numerical, and kernel-verified claims separately. No code should claim verified discovery unless the Lean kernel and axiom audit both pass. See `docs/ROADMAP.md`.

Run `python -m unittest discover -s tests -v` and the `usf06` benchmark smoke test before opening a PR.
