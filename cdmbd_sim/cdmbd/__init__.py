"""
cdmbd-sim
=========
Computational demonstration of C-DMBD (Constrained Dynamic Markov Blanket
Detection) for requirement-steered product design.

Companion code for:
    Possati, L. M. (2026). Design, Cups, and Blankets:
    A Free-Energy-Principle-Based Approach to Product Design.
    University of Twente.

Quick start
-----------
>>> from cdmbd import generate, run_cdmbd, make_profile
>>> Y, _ = generate('tea')
>>> result = run_cdmbd(Y, make_profile('tea'))
>>> print(result.n_B, result.rho_bb)
"""

from cdmbd.simulator import generate, STYLES
from cdmbd.core import (
    Requirement,
    RequirementProfile,
    make_profile,
    run_cdmbd,
    ensemble_run,
    modal_result,
    compute_violations,
    gap_score,
    CDMBDResult,
)

__version__ = "0.1.0"
__author__  = "Luca M. Possati"
__email__   = "l.m.possati@utwente.nl"

__all__ = [
    # data generation
    "generate",
    "STYLES",
    # requirements
    "Requirement",
    "RequirementProfile",
    "make_profile",
    # algorithm
    "run_cdmbd",
    "ensemble_run",
    "modal_result",
    "compute_violations",
    "gap_score",
    # result type
    "CDMBDResult",
]
