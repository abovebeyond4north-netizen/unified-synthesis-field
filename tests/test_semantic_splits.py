"""Public evaluator fixtures; do not log target keys or run learning searches."""
import unittest

from usf06.core import Expr, SealedTask, expr, make_tasks
from usf06.semantic_splits import (
    PUBLIC_DEV3_SEEDS, make_semantic_splits, polynomial_key, verify_semantic_splits,
)


class SemanticSplitTests(unittest.TestCase):
    def test_exact_polynomial_identity_detects_structural_alias(self):
        x = Expr.var()
        a = Expr("add", (x,x))
        b = Expr("mul", (Expr.const(2),x))
        self.assertNotEqual(a.key(), b.key())
        self.assertEqual(polynomial_key(a), polynomial_key(b))
        square = Expr("mul", (Expr("add",(x,Expr.const(1))),)*2)
        explicit = expr("add", expr("add",expr("mul",x,x),
                                    expr("mul",Expr.const(2),x)),Expr.const(1))
        self.assertEqual(polynomial_key(square),polynomial_key(explicit))
        with self.assertRaisesRegex(ValueError,"semantic overlap: fit / validation"):
            verify_semantic_splits(
                [SealedTask("a","linear",a)], [SealedTask("b","linear",b)],
                [SealedTask("c","quadratic",square)])

    def test_old_dev2_fit_validation_overlap_reproduced_without_final_targets(self):
        counts=[]
        for seed in (78201,78202,78203):
            training,_=make_tasks(seed,26,0)
            fit={polynomial_key(t._target) for t in training[:20]}
            counts.append(sum(polynomial_key(t._target) in fit for t in training[20:]))
        self.assertEqual(counts,[1,1,1])

    def test_new_fixed_public_splits_are_disjoint_and_repeatable(self):
        self.assertEqual(PUBLIC_DEV3_SEEDS,(78301,78302,78303))
        for seed in PUBLIC_DEV3_SEEDS:
            with self.subTest(seed=seed):
                splits=make_semantic_splits(seed)
                self.assertEqual(splits,make_semantic_splits(seed))
                receipt=verify_semantic_splits(*splits)
                self.assertEqual(receipt["counts"],{"fit":20,"validation":6,"final":6})
                self.assertEqual(receipt["unique_polynomial_functions"],32)
                self.assertTrue(receipt["semantic_disjointness_checked"])
                self.assertFalse(receipt["protected_evaluation_established"])
                self.assertFalse(receipt["independent_provenance_authenticated"])
                self.assertEqual([t.family for t in splits[2]],["quadratic","cubic"]*3)
                self.assertNotIn("target",str(receipt))
                self.assertNotIn("coefficients",str(receipt))

    def test_invalid_sizes_and_insufficient_capacity(self):
        for seed in (True,-1,2**63):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_semantic_splits(seed)
        for sizes in ((0,6,6),(True,6,6),(32,32,6)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                make_semantic_splits(1,*sizes)

    def test_duplicate_within_split_and_bounded_ast_reject(self):
        x=Expr.var()
        f=SealedTask("f","linear",x)
        v=SealedTask("v","linear",expr("add",x,Expr.const(1)))
        t=SealedTask("t","quadratic",expr("mul",x,x))
        with self.assertRaisesRegex(ValueError,"semantic overlap: fit / fit"):
            verify_semantic_splits([f,SealedTask("f2","linear",x)],[v],[t])
        with self.assertRaisesRegex(ValueError,"task identity"):
            verify_semantic_splits([f],[SealedTask("f","linear",v._target)],[t])
        huge=x
        for _ in range(8):
            huge=Expr("mul",(huge,huge))
        with self.assertRaises(ValueError):
            polynomial_key(huge)


if __name__=="__main__":
    unittest.main()
