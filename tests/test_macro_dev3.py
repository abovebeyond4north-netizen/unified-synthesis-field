"""DEV-3 preflight and orchestration tests; no DEV-3 learning/scoring."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from usf06.core import Expr, SealedTask
from usf06.macro_dev3 import dev3_descriptor, preflight_dev3, run_dev3
from usf06.macro_utility import digest
from usf06.semantic_splits import make_semantic_splits


class Dev3ProtocolTests(unittest.TestCase):
    def test_preflight_deterministic_disjoint_and_no_search(self):
        with patch("usf06.macro_dev3.run_validation") as fit, patch(
            "usf06.macro_dev3.evaluate_frozen"
        ) as final:
            pre = preflight_dev3()
            self.assertEqual(pre, preflight_dev3())
            fit.assert_not_called()
            final.assert_not_called()
        self.assertEqual(pre["protocol"], dev3_descriptor())
        self.assertEqual(pre["protocol"]["development_seeds"], [78301,78302,78303])
        self.assertEqual(pre["protocol_sha256"], digest(pre["protocol"]))
        self.assertEqual(pre["preflight_sha256"], digest(
            {k:v for k,v in pre.items() if k!="preflight_sha256"}))
        self.assertEqual([s["unique_polynomial_functions"] for s in pre["split_checks"]], [32]*3)
        self.assertFalse(pre["search_performed"])
        self.assertFalse(pre["independent_review_established"])
        self.assertFalse(pre["milestone_claim_eligible"])

    def test_wrong_commitment_and_old_seed_fail_before_work(self):
        with TemporaryDirectory() as tmp, patch("usf06.macro_dev3.run_validation") as fit:
            for seed, hash_value in ((78301,"0"*64),(78201,preflight_dev3()["preflight_sha256"])):
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    run_dev3(seed,tmp,expected_preflight_sha256=hash_value)
            fit.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_generator_overlap_rejected_before_search(self):
        original = make_semantic_splits(78301)
        contaminated = (original[0], [original[0][0],*original[1][1:]], original[2])
        with patch("usf06.macro_dev3.make_semantic_splits", return_value=contaminated), patch(
            "usf06.macro_dev3.run_validation"
        ) as fit:
            with self.assertRaisesRegex(ValueError,"duplicate or invalid task identity"):
                preflight_dev3()
            fit.assert_not_called()

    def test_mocked_search_and_scores_exercise_real_freeze_and_fallback(self):
        # No actual DEV-3 search or target score is executed; only protocol
        # orchestration, validation receipt logic and persistence are real.
        h=preflight_dev3()["preflight_sha256"]
        events=[]
        def fit_search(task,budget,seed,mode,**kwargs):
            self.assertTrue(kwargs["train"])
            events.append("fit")
            return SimpleNamespace(proposals=6,attempts=0,
                selected=SimpleNamespace(candidate=Expr.var(),survived=False))
        with TemporaryDirectory() as tmp:
            frozen=Path(tmp)/"dev3-78301-validation.json"
            def final_search(task,budget,seed,mode,**kwargs):
                self.assertTrue(frozen.is_file())
                self.assertIsNone(kwargs["macro"])
                events.append("final")
                return SimpleNamespace(proposals=6,attempts=0,macro_attempts=0,
                    macro_proposals=0,selected=SimpleNamespace(candidate=Expr.var()))
            def score(task,candidate):
                events.append("score")
                return SimpleNamespace(exact_fraction=0.0)
            with patch("usf06.macro_validation.search",side_effect=fit_search), patch(
                "usf06.macro_final.search",side_effect=final_search
            ), patch.object(SealedTask,"holdout_score",score):
                report,result=run_dev3(78301,tmp,expected_preflight_sha256=h)
            self.assertEqual(events,["fit"]*20+["final"]*18+["score"]*18)
            self.assertEqual(report["dev3_preflight"]["preflight_sha256"],h)
            self.assertEqual(report["semantic_split_receipt"]["unique_polynomial_functions"],32)
            self.assertTrue(result["macro_arm_fallback"])
            self.assertEqual(result["decision_reason"],"REJECT_NO_MACRO")
            self.assertEqual(json.loads(frozen.read_text())["validation_report"]["dev3_preflight"],
                             report["dev3_preflight"])
            with self.assertRaisesRegex(ValueError,"existing DEV-3 artifact"):
                run_dev3(78301,tmp,expected_preflight_sha256=h)

    def test_manifest_mismatch_refuses_final(self):
        h=preflight_dev3()["preflight_sha256"]
        bogus={"development_seed":78301,"manifest":{"fit_ids":[]}}
        with TemporaryDirectory() as tmp, patch(
            "usf06.macro_dev3.run_validation",return_value=bogus
        ), patch("usf06.macro_dev3.evaluate_frozen") as final:
            with self.assertRaisesRegex(ValueError,"manifest mismatch"):
                run_dev3(78301,tmp,expected_preflight_sha256=h)
            final.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()),[])


if __name__=="__main__":
    unittest.main()
