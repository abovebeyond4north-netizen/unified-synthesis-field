"""Synthetic controls for B2.5/B2.6. No frozen DEV-2 trial is executed."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import copy
import json
import unittest
from unittest.mock import patch

from usf06.core import Expr, SealedTask, make_tasks
from usf06.macro_final import freeze_validation, evaluate_frozen, run_dev2
from usf06.macro_utility import decide_utility
from usf06.macro_validation import run_validation
from usf06.search import search


def ast(n):
    v = ["add", ["x"], ["const", n]]
    return ["mul", ["add", v, v], ["sub", v, ["neg", v]]]


def controlled_report(admit):
    fit, val = make_tasks(78421, 4, 2)
    _, final = make_tasks(78422, 6, 2)
    indices = {t.task_id: i for i, t in enumerate(fit)}

    def controlled_search(task, budget, seed, mode, **kw):
        if kw.get("train"):
            return SimpleNamespace(proposals=budget, attempts=budget*2,
                selected=SimpleNamespace(candidate=Expr.from_serial(ast(indices[task.task_id]+1)),
                                         survived=True))
        return search(task, budget, seed, mode, **kw)

    with patch("usf06.macro_validation.search", side_effect=controlled_search):
        report = run_validation(fit, val, [t.task_id for t in final], seed=78421,
                                budget=60, validation_offsets=(200000,))
    assert report["candidate"] is not None
    # Mechanism fixture ONLY: set synthetic validation outcomes on actual
    # receipt shapes; never claim these are measured learning gains.
    for row in report["validation_rows"]:
        row["exact_success"] = int(admit and row["arm"] == "macro")
        if row["arm"] == "macro":
            row["macro_attempts"] = 10
            row["macro_proposals"] = 3
    for row, score in zip(report["validation_rows"], report["validation_scores"]):
        score["exact_fraction"] = float(row["exact_success"])
    report["decision"] = decide_utility(report["manifest"], report["candidate"],
                                        report["validation_rows"])
    return report, final


class FinalFreezeTests(unittest.TestCase):
    def test_admit_and_reject_preserve_phase_order_and_extension_choice(self):
        for admit in (True, False):
            with self.subTest(admit=admit), TemporaryDirectory() as tmp:
                report, final = controlled_report(admit)
                self.assertEqual(report["decision"]["admitted"], admit)
                frozen, out = Path(tmp)/"frozen.json", Path(tmp)/"final.json"
                h = freeze_validation(report, frozen)
                events = []
                original_score = SealedTask.holdout_score
                def spy(task, budget, seed, mode, **kw):
                    self.assertTrue(frozen.is_file())
                    events.append(("search", kw["macro"] is not None, kw["known_composition"]))
                    return search(task, budget, seed, mode, **kw)
                def score(task, candidate):
                    events.append(("score",))
                    return original_score(task, candidate)
                with patch("usf06.macro_final.search", side_effect=spy), patch.object(
                    SealedTask, "holdout_score", score
                ):
                    result = evaluate_frozen(frozen, h, final, out)
                self.assertEqual(events, [
                    ("search",False,False), ("search",False,True), ("search",admit,False)
                ]*2+[("score",)]*6)
                self.assertEqual(result["macro_admitted"], admit)
                self.assertEqual(result["macro_arm_fallback"], not admit)
                self.assertEqual(result, json.loads(out.read_text()))
                self.assertEqual(json.loads(frozen.read_text())["validation_report"]["candidate"],
                                 report["candidate"])
                self.assertFalse(result["milestone_claim_eligible"])
                if not admit:
                    for k in ("proposals", "attempts", "exact_success"):
                        self.assertEqual(result["totals"]["macro"][k], result["totals"]["primitive"][k])
                    self.assertEqual(result["totals"]["macro"]["macro_attempts"],0)

    def test_real_no_macro_receipt_and_final_fallback(self):
        fit, final = make_tasks(78425, 1, 2)
        _, val = make_tasks(78425, 3, 2)
        # IDs are disjoint; final and validation targets are public fixtures.
        report = run_validation(fit, val, [t.task_id for t in final], seed=78425, budget=12)
        self.assertEqual(report["decision"]["reason"], "REJECT_NO_MACRO")
        with TemporaryDirectory() as tmp:
            p, q = Path(tmp)/"freeze.json", Path(tmp)/"out.json"
            h = freeze_validation(report, p)
            result = evaluate_frozen(p, h, final, q)
            self.assertTrue(result["macro_arm_fallback"])
            self.assertEqual(result["decision_reason"], "REJECT_NO_MACRO")
            self.assertEqual(result["totals"]["macro"]["macro_proposals"], 0)

    def test_forged_decision_and_provenance_rejected_before_freeze(self):
        original, _ = controlled_report(False)
        for variant in ("decision", "source", "fit", "receipt", "cap", "score"):
            report = copy.deepcopy(original)
            if variant == "decision":
                report["decision"]["admitted"] = True
            elif variant == "source":
                report["source_hashes"]["core.py"] = "0"*64
            elif variant == "fit":
                report["training"][0]["challenge_survived"] = False
            elif variant == "receipt":
                report["validation_rows"].pop()
            elif variant == "score":
                report["validation_scores"][0]["exact_fraction"] = 1.0
            else:
                report["validation_rows"][0]["proposal_cap"] += 1
            with self.subTest(variant=variant), TemporaryDirectory() as tmp:
                p = Path(tmp)/"freeze.json"
                with self.assertRaises(ValueError):
                    freeze_validation(report, p)
                self.assertFalse(p.exists())

    def test_byte_tamper_wrong_ids_and_no_overwrite(self):
        report, final = controlled_report(False)
        with TemporaryDirectory() as tmp:
            p,q = Path(tmp)/"freeze.json",Path(tmp)/"out.json"
            h = freeze_validation(report,p)
            with self.assertRaises(FileExistsError):
                freeze_validation(report,p)
            with patch("usf06.macro_final.search") as spy:
                with self.assertRaisesRegex(ValueError,"IDs/order"):
                    evaluate_frozen(p,h,list(reversed(final)),q)
                p.write_bytes(p.read_bytes()+b" ")
                with self.assertRaisesRegex(ValueError,"commitment mismatch"):
                    evaluate_frozen(p,h,final,q)
                spy.assert_not_called()
            self.assertFalse(q.exists())

    def test_detect_changed_freeze_during_evaluation(self):
        report,final=controlled_report(False)
        with TemporaryDirectory() as tmp:
            p,q=Path(tmp)/"freeze.json",Path(tmp)/"out.json"
            h=freeze_validation(report,p)
            original_score=SealedTask.holdout_score
            def changed(task,candidate):
                p.write_bytes(p.read_bytes()+b" ")
                return original_score(task,candidate)
            with patch.object(SealedTask,"holdout_score",changed), self.assertRaisesRegex(ValueError,"commitment mismatch"):
                evaluate_frozen(p,h,final,q)
            self.assertFalse(q.exists())

    def test_wrapper_requires_frozen_seed(self):
        with self.assertRaises(ValueError):
            run_dev2(78124,"unused")


if __name__=="__main__":
    unittest.main()
