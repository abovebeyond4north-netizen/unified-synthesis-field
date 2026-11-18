import unittest
from usf06.macro_induction import induce_root_macro


# A repeated variable subtree and constants, with no supplied macro template.
def tree(n):
    v = ("add", ("x",), ("const", n))
    return ("mul", ("add", v, v), ("sub", v, ("neg", v)))


class RootMacroTests(unittest.TestCase):
    def test_induction_expansion_and_determinism(self):
        training = [(str(i), tree(i)) for i in range(1, 5)]
        macro = induce_root_macro(training)
        self.assertIsNotNone(macro)
        self.assertEqual(macro, induce_root_macro(reversed(training)))
        self.assertGreater(macro.training_node_savings, 0)
        self.assertEqual(macro.source_ids, ("1", "2", "3", "4"))
        for i in (1, 2, 3, 4, 9, -7):
            self.assertEqual(macro.expand(macro.arguments(tree(i))), tree(i))
        self.assertIsNone(macro.arguments(("add", ("x",), ("const", 50))))

    def test_alias_and_duplicate_sources_do_not_qualify(self):
        self.assertIsNone(induce_root_macro([(str(i), ("add", ("x",), ("const", i))) for i in range(10)]))
        self.assertIsNone(induce_root_macro([(str(i), tree(1)) for i in range(10)]))
        with self.assertRaises(ValueError):
            induce_root_macro([("a", tree(1)), ("a", tree(2))])

    def test_repeated_holes_require_consistent_bindings(self):
        macro = induce_root_macro([(str(i), tree(i)) for i in range(1, 5)])
        t = tree(9)
        mismatch = ("mul", t[1], tree(8)[2])
        self.assertIsNone(macro.arguments(mismatch))
        with self.assertRaises(ValueError):
            macro.expand(())

    def test_invalid_grammar_and_limits(self):
        for t in (("hole", 0), ("add", ("x",)), ("const", True), ("const", 1001)):
            with self.subTest(t=t), self.assertRaises(ValueError):
                induce_root_macro([("a", t)])
        with self.assertRaises(ValueError):
            induce_root_macro([(str(i), tree(i)) for i in range(33)])
        large = ("x",)
        for _ in range(12):
            large = ("add", large, large)
        with self.assertRaisesRegex(ValueError, "AST too large"):
            induce_root_macro([("a", large)])
