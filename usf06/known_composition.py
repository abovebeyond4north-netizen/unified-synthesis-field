"""Predeclared known-composition baseline for public development only.

The catalog is the full Cartesian product of the eight original primitive
mutations. Sampling uniformly from its ordered pairs is deliberately naive:
no corpus, learned representation, validation score, or test outcome is used.
One extra action slot chooses a pair and constructs its two primitive steps.
"""
from __future__ import annotations

from itertools import product

from .macro_utility import RULE, digest

PROTOCOL = "USF-MACRO-UTILITY-PUBLIC-DEV-2"
BASE_ACTIONS = ("add_x", "sub_x", "mul_x", "add_one", "sub_one", "mul_two", "negate", "square")
KNOWN_PAIRS = tuple(product(BASE_ACTIONS, repeat=2))
DEVELOPMENT_SEEDS = (78201, 78202, 78203)
FIT_TASKS = 20
VALIDATION_TASKS = 6
FINAL_TASKS = 6
PROPOSAL_CAP = 160
ATTEMPT_CAP = 1600
VALIDATION_SEED_OFFSETS = (200_000, 300_000)
FINAL_SEED_OFFSET = 400_000


def comparator_descriptors():
    common = {
        "base_actions": list(BASE_ACTIONS),
        "search_mode": "evolution",
        "parent_selection": "original_evolution",
        "initial_candidates": 6,
        "candidate_max_nodes": 23,
        "candidate_max_depth": 9,
        "scoring": "original_primitive_ast",
    }
    return {
        "primitive": {**common, "extra_action": None},
        "known": {**common, "extra_action": "known_composition",
                  "catalog": [list(pair) for pair in KNOWN_PAIRS],
                  "catalog_selection": "uniform_ordered_pair_with_replacement",
                  "construction": "two_sequential_primitive_mutations"},
        "macro": {**common, "extra_action": "derived_macro",
                  "construction": "frozen_fit_induced_macro_normalized",
                  "extra_arguments": "uniform_current_pool"},
    }


def comparator_digests():
    return {arm: digest(spec) for arm, spec in comparator_descriptors().items()}


def protocol_descriptor():
    """Public protocol commitment; no statement about unseen independent tests."""
    return {
        "protocol": PROTOCOL, "rule": RULE,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "split": {"fit": FIT_TASKS, "validation": VALIDATION_TASKS, "final": FINAL_TASKS},
        "task_generation": "make_tasks(seed,26,6); first20 fit; next6 validation; last6 final",
        "validation_seed_offsets": list(VALIDATION_SEED_OFFSETS),
        "final_seed_offset": FINAL_SEED_OFFSET,
        "fit_search_seed": "development_seed + fit_index * 101 + 7",
        "paired_search_seed": "phase_seed + task_index * 101",
        "proposal_cap": PROPOSAL_CAP, "attempt_cap": ATTEMPT_CAP,
        "decision_metric": "binary_exact_success_count_on_complete_paired_validation_matrix",
        "ties": "reject_against_either_comparator",
        "require_accepted_macro_proposal": True,
        "no_macro": "reject_and_retain",
        "comparator_sha256": comparator_digests(),
        "compute_equivalence_established": False,
        "independent_evaluation": False,
    }
