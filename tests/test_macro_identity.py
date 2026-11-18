"""Regression: macro and ordinary construction share candidate identity."""
import unittest
from unittest.mock import patch

from usf06.core import Expr
from usf06.macro_induction import induce_root_macro
from usf06.search import search


class SquareFixture:
    task_id = "synthetic-square"

    def training_samples(self):
        return ((-2, 4), (-1, 1), (0, 0), (1, 1), (2, 4))

    def falsify(self, candidate):
        return None if all(candidate.evaluate(x) == x * x for x in (-3, 3)) else {"mismatch": True}


class ForcedChoices:
    """Square x once, then expand the induced square template at zero."""
    def __init__(self, seed):
        self.actions = 0

    def random(self):
        return 0.0

    def choice(self, items):
        if isinstance(items[0], str):
            self.actions += 1
            return "square" if self.actions == 1 else "derived_macro"
        wanted = Expr.var() if self.actions == 0 else Expr.const(0)
        return next(item for item in items if item.candidate == wanted)


class MacroIdentityTests(unittest.TestCase):
    def test_macro_equivalent_to_seen_primitive_is_not_a_new_proposal(self):
        macro = induce_root_macro([
            (str(i), ("mul", ("add", ("x",), ("const", i)),
                      ("add", ("x",), ("const", i))))
            for i in range(1, 5)
        ])
        self.assertIsNotNone(macro)
        with patch("usf06.search.random.Random", ForcedChoices):
            result = search(SquareFixture(), 8, 1, "evolution", macro=macro)
        self.assertEqual(result.proposals, 7)
        self.assertEqual(result.attempts, 80)
        self.assertEqual(result.macro_attempts, 79)
        self.assertEqual(result.macro_proposals, 0)
        self.assertEqual(result.selected.candidate.evaluate(3), 9)


    def test_normalization_uses_primitive_order_and_constructor_semantics(self):
        from usf06.core import expr
        from usf06.search import _normalize_macro_ast

        raw = Expr.from_serial(["add", ["x"], ["const", 3]])
        normalized = _normalize_macro_ast(raw)
        self.assertEqual(normalized, expr("add", Expr.var(), Expr.const(3)))
        self.assertEqual(_normalize_macro_ast(normalized), normalized)
        for x in (-10000, -1, 0, 1, 10000):
            self.assertEqual(raw.evaluate(x), normalized.evaluate(x))

        # Construction semantics can erase an overflowing intermediate.
        # The resulting candidate is scored as the constructed AST, not
        # certified as equivalent to raw bounded evaluation.
        square = ["mul", ["x"], ["x"]]
        raw = Expr.from_serial(["mul", ["const", 0], ["mul", square, square]])
        with self.assertRaises(OverflowError):
            raw.evaluate(10000)
        normalized = _normalize_macro_ast(raw)
        self.assertEqual(normalized, Expr.const(0))
        self.assertEqual(normalized.evaluate(10000), 0)


if __name__ == "__main__":
    unittest.main()
