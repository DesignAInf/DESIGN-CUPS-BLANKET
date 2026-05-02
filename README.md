
# cdmbd-sim

**Computational demonstration of C-DMBD for requirement-steered product design**

Companion code for:

> Possati, L. M. (2026). *Design, Cups, and Blankets: A Free-Energy-Principle-Based
> Approach to Product Design*. University of Twente.

[![CI](https://github.com/possati/cdmbd-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/possati/cdmbd-sim/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this code does

The paper proposes **C-DMBD** (Constrained Dynamic Markov Blanket Detection): an
algorithm that infers the *interface type* of a product (e.g. espresso cup, travel
mug) from sensor data and functional requirements, using the Free Energy Principle.

This repository provides a standalone computational demonstration of the three
phenomena identified in the paper, plus a closed design loop.

| Experiment | Claim demonstrated |
|---|---|
| **P1** – Intra-family navigation | Same physical topology `\|B\|`, different functional mode (ρ, λ*) under different requirement profiles |
| **P2** – Family transition | Continuous requirement scan → discontinuous jump in `\|B\|` (phase transition of interface type) |
| **P3** – Ontological disambiguation | When data are spatially flat (gap Δ ≈ 0), requirements — not data — determine the interface type |
| **Design loop** | DKL(p̃_user ‖ p_cup) decreases across iterations as the designer's model converges toward user preferences |

---

## Mathematical core

**State-space model** (linear-Gaussian with Markov blanket structure):

```
x(t) = A x(t-1) + η,      η ~ N(0, Q)
y_i(t) = C_ωᵢ x(t) + d_ωᵢ + εᵢ,    εᵢ ~ N(0, σ²I)
ωᵢ ∈ {S, B, Z}
```

**MB zero constraints:**  `A[z,s] = 0`,  `A[s,z] = 0`

**Stationary blanket mean** (Eq. 4):
```
μ̂_b = C_b (I − A_bb)⁻¹ b̄ + d_b
```

**Requirement violation** (Eq. 5):
```
v_k(ω, Θ) = w_k max(0, |φ_k(ω,Θ) − τ_k| − ε_k)
```

**Augmented ELBO** (Eq. 9):
```
L_aug = L_ELBO − Σ_k λ_k v_k(ω, Θ)
```

**Dual ascent** (Eq. 11):
```
λ_k ← clip(λ_k + α_λ v_k, 0, λ_max)
```

**Optimal design** (Def. 5):
```
Θ** = argmin_{Θ ∈ F(B)} DKL(p̃_user ‖ p_cup(y_b | Θ))
```

---

## Installation

```bash
git clone https://github.com/possati/cdmbd-sim.git
cd cdmbd-sim
pip install -e .
```

**Requirements:** Python ≥ 3.10, numpy, scipy, matplotlib (all installed automatically).

---

## Quick start

### Run all experiments

```bash
cdmbd-run                       # all four experiments
cdmbd-run --p1                  # P1 only (intra-family navigation)
cdmbd-run --p2                  # P2 only (family transition)
cdmbd-run --p3                  # P3 only (ontological disambiguation)
cdmbd-run --loop                # design loop only
cdmbd-run --out ./my_figures    # custom output directory
```

Figures are saved as both PDF and PNG in `./figures/`.

### Python API

```python
from cdmbd import generate, run_cdmbd, make_profile, ensemble_run, modal_result

# 1. Generate synthetic cup data  (T=50, N=30 nodes, D=7 channels)
Y, partition_true = generate('tea')

# 2. Run C-DMBD with a requirement profile
profile = make_profile('tea')
result  = run_cdmbd(Y, profile, n_iter=50)

print(result.n_B)       # blanket size |B|
print(result.rho_bb)    # spectral radius ρ(A_bb)
print(result.lambda_)   # converged Lagrange multipliers λ*

# 3. Ensemble for robustness
runs = ensemble_run(Y, profile, n_runs=6)
best = modal_result(runs)
```

---

## Repository structure

```
cdmbd-sim/
├── cdmbd/
│   ├── __init__.py        Public API
│   ├── simulator.py       Synthetic thermal cup data generator
│   ├── core.py            C-DMBD algorithm (E-step, M-step, dual ascent)
│   ├── phenomena.py       P1, P2, P3 experiment functions
│   ├── design_loop.py     Closed design loop (3 users × 3 styles × 3 iterations)
│   └── cli.py             Command-line entry point (cdmbd-run)
├── tests/
│   └── test_cdmbd.py      Unit tests (pytest)
├── examples/
│   └── minimal_example.py Minimal working example
├── .github/
│   └── workflows/ci.yml   GitHub Actions CI
├── pyproject.toml
├── LICENSE                MIT
└── README.md
```

---

## Observation channels

| Index | Channel | Unit |
|---|---|---|
| 0 | T_inner | °C |
| 1 | T_outer | °C |
| 2 | grad_T | °C/m |
| 3 | heat flux | W/m² |
| 4 | radius | m |
| 5 | curvature | 1/m |
| 6 | wall thickness | m |

---

## Cup styles

| Style | Volume | T_inner | T_outer (sw) | T_outer (dw) | ρ_dyn |
|---|---|---|---|---|---|
| espresso | 60 ml | 90 °C | 86 °C | 30.9 °C | 0.47 |
| tea | 250 ml | 85 °C | 83 °C | 31.0 °C | 0.51 |
| travel_mug | 350 ml | 80 °C | 86 °C | 30.9 °C | 0.52 |

`sw` = single-wall,  `dw` = double-wall.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Implementation notes

This is a proof-of-concept implementation. The E-step uses a simplified node
assignment (spatial prior + augmented marginal likelihood) rather than full
Kalman-HMM smoothing. The M-step uses gradient ascent on the augmented ELBO.

The key mechanism — requirement violations depending on the *stationary statistics*
of the fitted model, not on raw data — is implemented correctly and drives both
intra-family navigation (P1) and family transition (P2).

For the full VBEM implementation with Kalman smoothing, see:
Beck, D. & Ramstead, M. J. D. (2025). *Dynamic Markov Blanket Detection*.
arXiv:2502.21217.

---

## Citation

```bibtex
@techreport{possati2026cups,
  title   = {Design, Cups, and Blankets: A Free-Energy-Principle-Based
             Approach to Product Design},
  author  = {Possati, Luca M.},
  year    = {2026},
  institution = {University of Twente},
}
```

---

## Author

**Luca M. Possati**  
Department of Design, Production and Management  
University of Twente  
l.m.possati@utwente.nl
