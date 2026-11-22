"""A2.7: real-process public four-arm comparison, not concealed evaluation.

Uses only the repository's intentionally public synthetic worker fixture.
This does NOT establish hostile-process isolation, independent provenance,
matched actual compute, autonomous invention, or cross-domain transfer.
"""
import json
import unittest

from usf06.evaluator_supervisor import DEFAULT_ARMS, PublicSupervisor
from usf06.search import AdaptivePolicy, search


BUDGET = 32
CHALLENGE_CAP = 48
SEEDS = (42, 97)


def run_public_comparison(seed):
    """Execute real child processes, freeze all arms, then release final scores."""
    candidates = {}
    policies = {}
    summary = {}
    with PublicSupervisor(
        challenge_cap=CHALLENGE_CAP,
        session_timeout=90.0,
    ) as supervisor:
        for arm in DEFAULT_ARMS:
            task = supervisor.open_arm(arm)
            mode = "adaptive" if arm in ("cold_adaptive", "adaptive") else arm
            policy = AdaptivePolicy() if mode == "adaptive" else None
            result = search(task, BUDGET, seed, mode, policy=policy, train=False)
            candidates[arm] = result.selected.candidate
            policies[arm] = policy.snapshot() if policy is not None else None

            receipt = supervisor.freeze_arm(arm, candidates[arm], policies[arm])
            learner = receipt["learner"]
            assert task.closed
            assert learner["observations"] == 1
            assert learner["frames"] == 1 + learner["challenges"]
            assert learner["challenges"] == task.challenge_requests
            assert learner["request_bytes"] == task.request_bytes
            assert 0 < learner["request_bytes"] <= learner["frames"] * 65536
            assert 0 <= learner["challenges"] <= CHALLENGE_CAP
            assert receipt["final_queries"] == 0
            summary[arm] = {
                "candidate_ast": candidates[arm].serial(),
                "proposals": result.proposals,
                "mutation_attempts": result.attempts,
                "training_survivors": result.survived_count,
                "observation_frames": learner["observations"],
                "challenge_requests": learner["challenges"],
                "learner_request_frames": learner["frames"],
                "learner_request_bytes": learner["request_bytes"],
            }

        assert not supervisor._released
        assert set(supervisor._frozen) == set(DEFAULT_ARMS)
        scored = supervisor.score_all(candidates, policies)
        assert supervisor._released
        assert not scored["independent_provenance_authenticated"]
        assert not scored["matched_actual_compute_established"]
        assert not scored["milestone_claim_eligible"]
        assert set(scored["final_receipts"]) == set(DEFAULT_ARMS)
        for arm in DEFAULT_ARMS:
            final = scored["final_receipts"][arm]
            assert final["final_queries"] == 1
            assert final["barrier_sha256"] == scored["barrier_sha256"]
            assert final["exact_success"] in (0, 1)
            assert 0 <= final["exact_fraction"] <= 1
            assert (final["mae"] == 0) == (final["exact_success"] == 1)
            summary[arm].update({
                "exact_success": final["exact_success"],
                "exact_fraction": final["exact_fraction"],
                "mae": final["mae"],
                "final_queries": final["final_queries"],
            })

    # Same search implementation and freshly initialized policy in both
    # adaptive arms. Their equality is a NEGATIVE learning control, not gain.
    assert summary["cold_adaptive"] == summary["adaptive"]
    return summary


class PublicEvaluatorE2EComparisonTests(unittest.TestCase):
    def test_four_arm_reproducible_public_comparison(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                first = run_public_comparison(seed)
                replay = run_public_comparison(seed)
                self.assertEqual(first, replay)
                self.assertEqual(set(first), set(DEFAULT_ARMS))
                for arm, record in first.items():
                    with self.subTest(seed=seed, arm=arm):
                        self.assertLessEqual(record["proposals"], BUDGET)
                        self.assertLessEqual(
                            record["challenge_requests"], record["proposals"]
                        )
                        self.assertGreater(record["learner_request_bytes"], 0)
                self.assertEqual(
                    first["adaptive"]["exact_success"]
                    - first["cold_adaptive"]["exact_success"],
                    0,
                )
                print("USF_A2_PUBLIC_FOUR_ARM_COMPARISON " + json.dumps({
                    "seed": seed,
                    "budget_per_arm": BUDGET,
                    "challenge_cap": CHALLENGE_CAP,
                    "source": "fixed_public_fixture",
                    "training_applied": False,
                    "independent_evaluation": False,
                    "matched_actual_compute_established": False,
                    "milestone_claim_eligible": False,
                    "arms": first,
                }, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
