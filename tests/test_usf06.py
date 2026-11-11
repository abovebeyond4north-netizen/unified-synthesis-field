import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from usf06.core import Expr, SealedTask, expr, fit, hypothesis_bits, make_tasks, sha256, universal_integer_bits
from usf06.ledger import Ledger
from usf06.logos import LogosGate, build_lean, parse_axiom_output
from usf06.pareto import FrontierItem, VerifiedParetoArchive, dominates
from usf06.search import AdaptivePolicy, search
from usf06.benchmark import run_benchmark


class ASTTests(unittest.TestCase):
    def test_roundtrip_and_evaluation(self):
        x = Expr.var()
        h = expr("add", expr("mul", x, x), Expr.const(2))
        self.assertEqual(h.evaluate(3), 11)
        self.assertEqual(Expr.from_serial(h.serial()), h)
        self.assertEqual(Expr.from_serial(h.serial()).key(), h.key())

    def test_commutative_normalization(self):
        x = Expr.var()
        self.assertEqual(expr("add", x, Expr.const(2)), expr("add", Expr.const(2), x))

    def test_unsafe_ast_rejected(self):
        with self.assertRaises(ValueError):
            Expr("exec", ())
        with self.assertRaises(ValueError):
            Expr.const(100000000)
        with self.assertRaises(ValueError):
            Expr("x", (Expr.var(),))

    def test_mdl_code_lengths(self):
        self.assertEqual(universal_integer_bits(0), 1)
        self.assertEqual(universal_integer_bits(-1), 3)
        self.assertEqual(universal_integer_bits(1), 3)
        h = expr("add", Expr.var(), Expr.const(1))
        self.assertGreater(hypothesis_bits(h), hypothesis_bits(Expr.var()))
        self.assertGreater(fit(h, [(x, x+1) for x in range(-3, 4)]).compression_gain_bits,
                           fit(Expr.const(0), [(x, x+1) for x in range(-3, 4)]).compression_gain_bits)

    def test_train_overfit_falsified(self):
        x = Expr.var()
        overfit = Expr.const(1)
        for r in (-2, -1, 0, 1, 2):
            overfit = expr("mul", overfit, expr("sub", x, Expr.const(r)))
        task = SealedTask("overfit", "stress", Expr.const(0))
        self.assertEqual(fit(overfit, task.training_samples()).exact_fraction, 1.0)
        self.assertIsNotNone(task.falsify(overfit))

    def test_sealed_data_splits(self):
        a,b = make_tasks(10, 4, 4)
        aa,bb = make_tasks(10, 4, 4)
        self.assertEqual(sha256([t.reveal() for t in a+b]), sha256([t.reveal() for t in aa+bb]))
        self.assertTrue(all(not (set(t.train_x) & set(t.holdout_x)) for t in a+b))
        self.assertTrue(all(not (set(t.challenge_x) & set(t.holdout_x)) for t in a+b))
        self.assertEqual({t.family for t in b}, {"quadratic", "cubic"})


class LedgerTests(unittest.TestCase):
    def test_immutability_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Ledger(Path(tmp)/"l.sqlite") as l:
                l.append("r1", "manifest", {"seed": 1})
                l.append("r1", "candidate", {"ast": "x"}, "C1")
                self.assertEqual(l.verify_integrity(), (True, 2))
                with self.assertRaises(sqlite3.IntegrityError):
                    l.db.execute("UPDATE ledger_events SET event_json='{}' WHERE sequence=1")
                with self.assertRaises(sqlite3.IntegrityError):
                    l.db.execute("DELETE FROM ledger_events WHERE sequence=1")
                self.assertEqual(l.verify_integrity(), (True, 2))

    def test_integrity_detects_untrusted_db_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Ledger(Path(tmp)/"l.sqlite") as l:
                l.append("r1", "manifest", {"seed": 1})
                l.db.execute("DROP TRIGGER prevent_ledger_update")
                l.db.execute("UPDATE ledger_events SET event_json=? WHERE sequence=1", ('{"seed":2}',))
                l.db.commit()
                self.assertEqual(l.verify_integrity(), (False, 1))


class ParetoTests(unittest.TestCase):
    def item(self, label, a, b, c, valid=True):
        return FrontierItem(label,a,b,c,valid,valid,valid)

    def test_weak_dominance_and_gate(self):
        p = self.item("p", 4, 0.8, 10)
        q = self.item("q", 4, 0.7, 10)
        self.assertTrue(dominates(p,q))  # equal in 2 dimensions, strictly better in 1
        self.assertFalse(dominates(p,p))
        archive = VerifiedParetoArchive()
        self.assertFalse(archive.admit(self.item("bad",10,1,1,False)))
        self.assertTrue(archive.admit(q))
        self.assertTrue(archive.admit(p))
        self.assertEqual([x.candidate_id for x in archive.items], ["p"])
        self.assertTrue(archive.admit(self.item("tradeoff",2,1.0,1)))
        self.assertEqual(len(archive.items),2)


class LogosTests(unittest.TestCase):
    def test_axiom_parser(self):
        s = "'usf_test' depends on axioms: [propext, Classical.choice, Quot.sound]"
        self.assertEqual(parse_axiom_output(s,"usf_test"), ("propext","Classical.choice","Quot.sound"))
        self.assertEqual(parse_axiom_output("'usf_test' does not depend on any axioms", "usf_test"), ())
        self.assertIsNone(parse_axiom_output("'usf_other' depends on axioms: []", "usf_test"))

    def test_formal_gate_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = LogosGate(tmp)
            r = gate.verify(Expr.var(), Expr.var(), "usf_test")
            self.assertEqual(r.status, "LEAN_UNAVAILABLE")
            self.assertFalse(r.kernel_verified)
            self.assertFalse(r.axiom_audit_pass)
            self.assertIn("#print axioms usf_test", r.lean_source)
            self.assertNotIn("sorry", r.lean_source)

    def test_compiler_guards_identifier(self):
        with self.assertRaises(ValueError):
            build_lean(Expr.var(), Expr.var(), "bad;unsafe")


class SearchAndBenchmarkTests(unittest.TestCase):
    def test_seeded_search_repeatable(self):
        task = make_tasks(1,2,2)[0][0]
        a = search(task, 65, 10, "evolution")
        b = search(task, 65, 10, "evolution")
        self.assertEqual(a.selected.candidate.key(), b.selected.candidate.key())
        self.assertEqual(a.proposals,b.proposals)
        self.assertLessEqual(a.proposals, 65)

    def test_frozen_policy_unchanged_on_test(self):
        task = make_tasks(1,2,2)[1][0]
        policy = AdaptivePolicy()
        initial = policy.snapshot()
        search(task, 45, 11, "adaptive", policy=policy, train=False)
        self.assertEqual(initial, policy.snapshot())

    def test_benchmark_and_reveal_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp)/"bench.sqlite")
            path = str(Path(tmp)/"report.json")
            report = run_benchmark(db,path,seed=7,budget=34,n_train=3,n_test=4)
            self.assertTrue(report["ledger_integrity"]["verified"])
            self.assertEqual(len(report["test_rows"]), 16)
            self.assertEqual(set(report["methods"]), {"random","evolution","cold_adaptive","adaptive"})
            self.assertFalse(report["claim_preflight"]["eligible"])
            self.assertIn("matched_wall_budget_enforced", report["claim_preflight"]["blocking_reasons"])
            self.assertIn("concealed_evaluator", report["claim_preflight"]["blocking_reasons"])
            self.assertTrue(Path(path).exists())
            self.assertTrue(all(r["lean_status"] != "KERNEL_VERIFIED" for r in report["test_rows"]))
            with Ledger(db) as l:
                events = l.events(report["run_id"])
                names = [x["event_type"] for x in events]
                self.assertLess(names.index("policy_frozen"), names.index("sealed_evaluation"))
                self.assertGreater(names.index("reveal_after_evaluation"), max(i for i,n in enumerate(names) if n == "sealed_evaluation"))
                reveal = next(e["payload"] for e in events if e["event_type"] == "reveal_after_evaluation")
                self.assertEqual(sha256(reveal["test"]),report["sealed_test_commitment"])
                self.assertTrue(l.verify_integrity()[0])


if __name__ == "__main__":
    unittest.main()

# Additional tests remain discoverable without relying on __main__ execution.
class ProvenanceTests(unittest.TestCase):
    def test_parent_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with Ledger(Path(d)/"test.sqlite") as l:
                with self.assertRaises(ValueError):
                    l.append("r", "candidate", {}, "child", "missing")
                l.append("r", "candidate", {}, "root")
                with self.assertRaises(ValueError):
                    l.append("r", "candidate", {}, "root")
                l.append("r", "candidate", {}, "child", "root")
                self.assertEqual(l.verify_integrity(), (True, 2))

    def test_timestamp_is_authenticated(self):
        with tempfile.TemporaryDirectory() as d:
            with Ledger(Path(d)/"test.sqlite") as l:
                l.append("r", "manifest", {"seed": 1})
                l.db.execute("DROP TRIGGER prevent_ledger_update")
                l.db.execute("UPDATE ledger_events SET created_utc='2000-01-01T00:00:00Z'")
                l.db.commit()
                self.assertFalse(l.verify_integrity()[0])

    def test_dynamic_falsifier_does_not_read_holdout(self):
        task = SealedTask("T-test", "toy", Expr.const(0))
        x = Expr.var()
        h = Expr.const(1)
        for r in task.train_x:
            h = expr("mul", h, expr("sub", x, Expr.const(r)))
        ce = task.falsify(h)
        self.assertIsNotNone(ce)
        self.assertNotIn(ce["x"], task.holdout_x)
        self.assertGreater(ce["max_absolute_error"], 0)
