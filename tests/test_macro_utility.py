"""B2.2 regression controls; all data synthetic, no evaluator or holdout."""
import copy
import unittest

from usf06.macro_utility import ARMS, RULE, decide_utility, digest


def fixture():
    m = {
        "rule": RULE, "fit_ids": ["fit-1", "fit-2"],
        "validation_ids": ["val-1", "val-2"], "test_ids": ["test-1"],
        "seeds": [1, 2], "proposal_cap": 20, "attempt_cap": 200,
        "architecture_sha256": "a"*64, "grammar_sha256": "b"*64,
        "comparators": {arm: str(i)*64 for i, arm in enumerate(ARMS)},
    }
    c = {"pattern": ["sub", ["sub", ["hole", 0], ["x"]], ["x"]],
         "arity": 1, "source_ids": ["fit-1", "fit-2"], "training_node_savings": 3}
    rows = []
    for seed in m["seeds"]:
        for task in m["validation_ids"]:
            for arm in ARMS:
                rows.append({
                    "seed": seed, "task_id": task, "arm": arm,
                    "manifest_sha256": digest(m), "candidate_sha256": digest(c),
                    "comparator_sha256": m["comparators"][arm],
                    "proposal_cap": 20, "attempt_cap": 200,
                    "exact_success": int(arm == "macro" or (arm == "known" and task == "val-1")),
                    "proposals": 20, "attempts": 40,
                    "macro_attempts": 10 if arm != "primitive" else 0,
                    "macro_proposals": 3 if arm != "primitive" else 0,
                    "search_ns": 100,
                })
    return m, c, rows


class UtilityGateTests(unittest.TestCase):
    def rejected(self, reason, m, c, rows):
        result = decide_utility(m, c, rows)
        self.assertEqual(result["reason"], reason)
        self.assertFalse(result["admitted"])
        self.assertFalse(result["milestone_claim_eligible"])
        return result

    def test_positive_and_order_invariance(self):
        m, c, rows = fixture()
        result = decide_utility(m, c, rows)
        self.assertEqual(result["reason"], "ADMIT_DEVELOPMENT_ONLY")
        self.assertTrue(result["admitted"])
        self.assertEqual(result["validation_totals"]["macro"]["exact_success"], 4)
        self.assertEqual(result["validation_totals"]["known"]["exact_success"], 2)
        self.assertEqual(result, decide_utility(m, c, list(reversed(rows))))
        self.assertEqual(result["decision_sha256"],
                         digest({k: v for k, v in result.items() if k != "decision_sha256"}))
        for flag in ("milestone_claim_eligible", "independent_provenance_authenticated",
                     "matched_actual_compute_established"):
            self.assertFalse(result[flag])

    def test_tie_or_loss_against_either_arm(self):
        for arm in ("primitive", "known"):
            for macro_success in (0, 1):
                with self.subTest(arm=arm, macro_success=macro_success):
                    m, c, rows = fixture()
                    for row in rows:
                        if row["arm"] == arm:
                            row["exact_success"] = 1
                        elif row["arm"] == "macro":
                            row["exact_success"] = macro_success
                    self.rejected("REJECT_NONPOSITIVE_UTILITY", m, c, rows)

    def test_unused_and_no_macro(self):
        m, c, rows = fixture()
        for row in rows:
            if row["arm"] == "macro":
                row["macro_proposals"] = 0
        self.rejected("REJECT_UNUSED", m, c, rows)
        self.rejected("REJECT_NO_MACRO", m, None, [])
        self.rejected("REJECT_INVALID_EVIDENCE", m, None, rows)

    def test_incomplete_duplicate_and_unknown_pairing(self):
        for kind in ("missing", "duplicate", "test", "fit", "seed", "arm"):
            m, c, rows = fixture()
            with self.subTest(kind=kind):
                if kind == "missing":
                    rows.pop()
                elif kind == "duplicate":
                    rows[-1] = copy.deepcopy(rows[0])
                elif kind in ("test", "fit"):
                    rows[0]["task_id"] = kind + "-1"
                elif kind == "seed":
                    rows[0]["seed"] = 3
                else:
                    rows[0]["arm"] = "other"
                result = self.rejected("REJECT_INVALID_EVIDENCE", m, c, rows)
                self.assertIsNone(result["validation_totals"])

    def test_malformed_counts_and_commitments(self):
        for field, value in (
            ("exact_success", True), ("exact_success", 0.5),
            ("exact_success", float("nan")), ("proposals", 21),
            ("attempts", 13), ("attempts", 201), ("attempts", True),
            ("macro_attempts", 41), ("macro_proposals", 15),
            ("proposal_cap", 21), ("attempt_cap", 201),
            ("manifest_sha256", "f"*64), ("candidate_sha256", "f"*64),
            ("comparator_sha256", "f"*64), ("search_ns", -1),
            ("search_ns", 10**1000),
        ):
            with self.subTest(field=field, value=repr(value)[:80]):
                m, c, rows = fixture()
                rows[-1][field] = value
                self.rejected("REJECT_INVALID_EVIDENCE", m, c, rows)

    def test_primitive_macro_usage_and_inconsistent_work_partition(self):
        m, c, rows = fixture()
        rows[0]["macro_attempts"] = 1
        self.rejected("REJECT_INVALID_EVIDENCE", m, c, rows)
        m, c, rows = fixture()
        rows[-1]["macro_attempts"] = 40
        self.rejected("REJECT_INVALID_EVIDENCE", m, c, rows)

    def test_manifest_split_and_candidate_validation(self):
        for kind in ("overlap", "duplicate_seed", "seed_bool", "same_comparator",
                     "foreign_source", "duplicate_source", "bad_hole", "alias", "cap"):
            m, c, rows = fixture()
            with self.subTest(kind=kind):
                if kind == "overlap":
                    m["test_ids"] = m["fit_ids"][:]
                elif kind == "duplicate_seed":
                    m["seeds"] = [1, 1]
                elif kind == "seed_bool":
                    m["seeds"] = [True, 2]
                elif kind == "same_comparator":
                    m["comparators"]["known"] = m["comparators"]["primitive"]
                elif kind == "foreign_source":
                    c["source_ids"] = ["fit-1", "val-1"]
                elif kind == "duplicate_source":
                    c["source_ids"] = ["fit-1", "fit-1"]
                elif kind == "bad_hole":
                    c["pattern"][1][1] = ["hole", 1]
                elif kind == "alias":
                    c["pattern"] = ["sub", ["hole", 0], ["x"]]
                else:
                    m["attempt_cap"] = 201
                # Recommit altered inputs: semantic validation must still reject.
                for row in rows:
                    row["manifest_sha256"] = digest(m)
                    row["candidate_sha256"] = digest(c)
                self.rejected("REJECT_INVALID_EVIDENCE", m, c, rows)

    def test_unknown_fields_and_nonrecord_inputs(self):
        m, c, rows = fixture()
        rows[0]["holdout"] = 1
        self.rejected("REJECT_INVALID_EVIDENCE", m, c, rows)
        self.rejected("REJECT_INVALID_EVIDENCE", None, c, rows)
        self.rejected("REJECT_INVALID_EVIDENCE", m, c, None)

    def test_inputs_unchanged(self):
        m, c, rows = fixture()
        before = copy.deepcopy((m, c, rows))
        decide_utility(m, c, rows)
        self.assertEqual((m, c, rows), before)


if __name__ == "__main__":
    unittest.main()
