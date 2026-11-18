"""Execute the prospectively frozen DEV-2 public protocol and retain all outcomes.

Python-version repeats are software reproducibility controls, not independent
replicates. No acceptance threshold depends on an observed experimental gain.
"""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from usf06.known_composition import DEVELOPMENT_SEEDS, PROPOSAL_CAP, ATTEMPT_CAP, protocol_descriptor
from usf06.macro_final import run_dev2
from usf06.macro_utility import ARMS, digest


def stable(obj):
    """Drop descriptive timing and timing-dependent hashes for repeat checks."""
    if isinstance(obj, dict):
        return {k: stable(v) for k, v in obj.items()
                if not k.endswith("_ns") and k not in (
                    "decision_sha256", "receipts_sha256", "validation_commitment")}
    if isinstance(obj, list):
        return [stable(v) for v in obj]
    return obj


class Dev2EvidenceTests(unittest.TestCase):
    def test_frozen_public_dev2_all_seeds(self):
        summaries = []
        for seed in DEVELOPMENT_SEEDS:
            with TemporaryDirectory() as tmp:
                validation, result = run_dev2(seed, tmp)
                f = Path(tmp)/f"dev2-{seed}-validation.json"
                q = Path(tmp)/f"dev2-{seed}-final.json"
                self.assertEqual(result["validation_commitment"], hashlib.sha256(f.read_bytes()).hexdigest())
                self.assertEqual(json.loads(q.read_text()), result)
                self.assertEqual(validation["protocol_sha256"], digest(protocol_descriptor()))
                self.assertEqual(len(validation["training"]),20)
                self.assertEqual(len(result["rows"]),18)
                self.assertFalse(result["milestone_claim_eligible"])
                self.assertFalse(result["matched_actual_compute_established"])
                for row in validation["validation_rows"]+result["rows"]:
                    self.assertLessEqual(row["proposals"],PROPOSAL_CAP)
                    self.assertLessEqual(row["attempts"],ATTEMPT_CAP)
                    self.assertLessEqual(row["macro_proposals"],row["macro_attempts"])
                if result["macro_arm_fallback"]:
                    self.assertEqual(result["totals"]["macro"]["macro_attempts"],0)
                    for key in ("exact_success","proposals","attempts"):
                        self.assertEqual(result["totals"]["macro"][key],result["totals"]["primitive"][key])
                # Print complete local artifacts in bounded chunks, including
                # unsuccessful induction and rejection evidence.
                for phase, artifact in (("validation",f),("final",q)):
                    raw=artifact.read_text()
                    chunks=[raw[i:i+6000] for i in range(0,len(raw),6000)]
                    for j,chunk in enumerate(chunks):
                        print("USF_DEV2_ARTIFACT "+json.dumps({
                            "seed":seed,"phase":phase,"part":j,"parts":len(chunks),
                            "sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            "chunk":chunk},sort_keys=True),flush=True)
                summary={
                    "seed":seed,"candidate":validation["candidate"],
                    "surviving_fit_outputs":validation["surviving_training_observations"],
                    "decision":result["decision_reason"],"admitted":result["macro_admitted"],
                    "fit_totals":validation["fit_totals"],"cost_ns":validation["cost_ns"],
                    "validation_totals":validation["decision"]["validation_totals"],
                    "final_totals":result["totals"],
                    "final_paired":[{a:next(r["exact_success"] for r in result["rows"]
                                          if r["arm"]==a and r["task_id"]==task)
                                     for a in ARMS} for task in validation["manifest"]["test_ids"]],
                    "stable_receipt_sha256":digest(stable([validation,result])),
                }
                summaries.append(summary)
                print("USF_DEV2_SEED "+json.dumps(summary,sort_keys=True),flush=True)
        totals={a:{k:sum(s["final_totals"][a][k] for s in summaries)
                   for k in ("exact_success","proposals","attempts","macro_attempts","macro_proposals")}
                for a in ARMS}
        print("USF_DEV2_SUMMARY "+json.dumps({
            "protocol_sha256":digest(protocol_descriptor()),
            "seeds":list(DEVELOPMENT_SEEDS),"paired_final_tasks":18,
            "induced":sum(s["candidate"] is not None for s in summaries),
            "admitted":sum(s["admitted"] for s in summaries),
            "decisions":[s["decision"] for s in summaries],"final_totals":totals,
            "stable_receipts":[s["stable_receipt_sha256"] for s in summaries],
            "scientific_acceptance":False},sort_keys=True),flush=True)
        self.assertEqual(len(summaries),len(DEVELOPMENT_SEEDS))


if __name__=="__main__":
    unittest.main()
