"""Strict formal admission and conventional weak Pareto dominance."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontierItem:
    candidate_id: str
    compression_gain: float  # maximize
    transfer_score: float    # maximize
    compute_cost: float      # minimize
    logos_verified: bool
    axiom_audit_pass: bool
    falsification_survived: bool


def dominates(a: FrontierItem, b: FrontierItem) -> bool:
    aa = (a.compression_gain, a.transfer_score, -a.compute_cost)
    bb = (b.compression_gain, b.transfer_score, -b.compute_cost)
    return all(x >= y for x, y in zip(aa, bb)) and any(x > y for x, y in zip(aa, bb))


class VerifiedParetoArchive:
    def __init__(self):
        self.items: list[FrontierItem] = []

    def admit(self, proposal: FrontierItem) -> bool:
        if not (proposal.logos_verified and proposal.axiom_audit_pass and proposal.falsification_survived):
            return False
        if any(x.candidate_id == proposal.candidate_id or dominates(x, proposal) for x in self.items):
            return False
        self.items = [x for x in self.items if not dominates(proposal, x)]
        self.items.append(proposal)
        self.items.sort(key=lambda x: x.candidate_id)
        return True
