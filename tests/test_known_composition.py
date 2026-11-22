"""B2.3: deterministic, data-independent two-step known-construct control."""
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from usf06.core import Expr, make_tasks
from usf06.known_composition import (
    BASE_ACTIONS, KNOWN_PAIRS, DEVELOPMENT_SEEDS, FIT_TASKS, VALIDATION_TASKS,
    FINAL_TASKS, PROPOSAL_CAP, ATTEMPT_CAP, comparator_descriptors,
    comparator_digests, protocol_descriptor,
)
from usf06.ledger import Ledger
from usf06.macro_induction import induce_root_macro
from usf06.macro_utility import RULE, digest
from usf06.search import ACTIONS, mutate, search


class LinearFixture:
    task_id = "known-linear-synthetic"

    def training_samples(self):
        return tuple((x, 2 * (x + 1)) for x in (-2, -1, 0, 1, 2))

    def falsify(self, candidate):
        return None if all(candidate.evaluate(x) == 2 * (x + 1) for x in (-3, 3)) else {"mismatch": True}


class ForceKnownPair:
    def __init__(self, seed):
        pass

    def random(self):
        return 0.0

    def choice(self, items):
        if isinstance(items[0], str):
            return "known_composition"
        if isinstance(items[0], tuple):
            return ("add_one", "mul_two")
        return next(item for item in items if item.candidate == Expr.var())


class KnownCompositionTests(unittest.TestCase):
    def test_complete_frozen_catalog_and_protocol(self):
        self.assertEqual(ACTIONS, BASE_ACTIONS)
        self.assertEqual(KNOWN_PAIRS, tuple(product(ACTIONS, repeat=2)))
        self.assertEqual(len(set(KNOWN_PAIRS)), 64)
        self.assertEqual(DEVELOPMENT_SEEDS, (78201, 78202, 78203))
        self.assertEqual((FIT_TASKS, VALIDATION_TASKS, FINAL_TASKS), (20, 6, 6))
        self.assertEqual((PROPOSAL_CAP, ATTEMPT_CAP), (160, 1600))
        protocol = protocol_descriptor()
        self.assertEqual(protocol["rule"], RULE)
        self.assertEqual(protocol["ties"], "reject_against_either_comparator")
        self.assertFalse(protocol["compute_equivalence_established"])
        self.assertFalse(protocol["independent_evaluation"])
        descriptors = comparator_descriptors()
        self.assertEqual(protocol["comparator_sha256"],
                         {arm: digest(spec) for arm, spec in descriptors.items()})
        self.assertEqual(len(set(comparator_digests().values())), 3)
        descriptors["known"]["catalog"].clear()
        self.assertEqual(protocol, protocol_descriptor())

    def test_forced_known_pair_generates_and_records_one_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Ledger(Path(tmp) / "known.sqlite") as ledger:
                with patch("usf06.search.random.Random", ForceKnownPair):
                    result = search(LinearFixture(), 7, 1, "evolution",
                                    known_composition=True, ledger=ledger, run_id="known")
                events = ledger.events("known")
                self.assertEqual(events[-1]["payload"]["action"],
                                 "known_composition:add_one:mul_two")
                self.assertTrue(ledger.verify_integrity()[0])
        self.assertEqual(result.selected.candidate,
                         mutate(mutate(Expr.var(), "add_one"), "mul_two"))
        self.assertTrue(result.selected.survived)
        self.assertEqual((result.proposals, result.attempts, result.macro_attempts,
                          result.macro_proposals), (7, 1, 1, 1))

    def test_opt_in_and_matched_configured_bounds(self):
        task = make_tasks(78401, 2, 2)[1][0]
        plain = search(task, 80, 1234, "evolution")
        self.assertEqual(plain, search(task, 80, 1234, "evolution", known_composition=False))
        known = search(task, 80, 1234, "evolution", known_composition=True)
        self.assertEqual(known, search(task, 80, 1234, "evolution", known_composition=True))
        self.assertEqual((plain.macro_attempts, plain.macro_proposals), (0, 0))
        self.assertGreater(known.macro_proposals, 0)
        self.assertLessEqual(known.macro_proposals, known.macro_attempts)
        for r in (plain, known):
            self.assertLessEqual(r.proposals, 80)
            self.assertLessEqual(r.attempts, 800)
            self.assertLessEqual(r.selected.candidate.size(), 23)
            self.assertLessEqual(r.selected.candidate.depth(), 9)

    def test_reject_mixed_extensions_or_nonfrozen_modes(self):
        task = LinearFixture()
        macro = induce_root_macro([
            (str(i), ("mul", ("add", ("x",), ("const", i)),
                      ("add", ("x",), ("const", i))))
            for i in range(1, 5)
        ])
        self.assertIsNotNone(macro)
        with self.assertRaises(ValueError):
            search(task, 10, 1, "evolution", macro=macro, known_composition=True)
        with self.assertRaises(ValueError):
            search(task, 10, 1, "evolution", train=True, known_composition=True)
        with self.assertRaises(ValueError):
            search(task, 10, 1, "random", known_composition=True)
        with self.assertRaises(ValueError):
            search(task, 10, 1, "evolution", known_composition=1)


if __name__ == "__main__":
    unittest.main()
