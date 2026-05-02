"""
simulator.py
------------
Synthetic thermal cup data generator.

Each cup is discretized into N nodes along the height axis.
Observation channels (D=7):
  0: T_inner   – inner surface temperature proxy
  1: T_outer   – outer surface temperature
  2: grad_T    – temperature gradient across wall
  3: flux      – heat flux
  4: radius    – node radius
  5: curvature – 1/radius
  6: wall_w    – wall thickness

Partition labels: 0=S (environment), 1=B (blanket/wall), 2=Z (interior)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ─── Style registry ───────────────────────────────────────────────────────────

@dataclass
class _Style:
    volume_ml:  float   # liquid volume (ml)
    T_inner:    float   # inner liquid temperature (°C)
    T_outer_sw: float   # outer surface temp, single-wall (°C)
    T_outer_dw: float   # outer surface temp, double-wall (°C)
    rho_dyn:    float   # target spectral radius of blanket dynamics
    wall_mm:    float   # wall thickness (mm)
    n_B_true:   int     # true blanket node count (for N=30)
    n_S_true:   int = 3 # true environment node count


STYLES: dict = {
    'espresso':   _Style(60,  90.0, 86.0, 30.9, 0.47, 3.0, 11),
    'tea':        _Style(250, 85.0, 83.0, 31.0, 0.51, 4.0,  6),
    'travel_mug': _Style(350, 80.0, 86.0, 30.9, 0.52, 8.0,  6),
}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _make_true_A(ds: int, db: int, dz: int, rho: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Random dynamics matrix A with Markov-blanket zero structure.
    Zeros:  A[z_idx, s_idx] = 0,  A[s_idx, z_idx] = 0
    Scaled so spectral radius ≈ rho.
    """
    d = ds + db + dz
    A = rng.normal(0, 0.25, (d, d))
    A[ds+db:, :ds] = 0   # Azs = 0
    A[:ds, ds+db:] = 0   # Asz = 0
    ev_max = np.max(np.abs(np.linalg.eigvals(A)))
    if ev_max > 1e-9:
        A *= rho * 0.90 / ev_max
    return A


# ─── Public API ───────────────────────────────────────────────────────────────

def generate(
    style: str,
    N: int = 30,
    T: int = 50,
    D: int = 7,
    double_wall: bool = False,
    flat_data: bool = False,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic cup observations.

    Parameters
    ----------
    style       : 'espresso' | 'tea' | 'travel_mug'
    flat_data   : if True, generate spatially uniform data (P3 demo – low gap)
    noise_scale : observation noise multiplier

    Returns
    -------
    Y              : (T, N, D) float64 – observation tensor
    partition_true : (N,)      int     – {0=S, 1=B, 2=Z}
    """
    rng = np.random.default_rng(seed)
    sp  = STYLES[style]

    T_out = sp.T_outer_dw if double_wall else sp.T_outer_sw

    # True partition  ─  S nodes first, then B (wall), then Z (interior)
    nS, nB = sp.n_S_true, sp.n_B_true
    nZ = N - nS - nB
    if nZ <= 0:
        raise ValueError(f"N={N} too small for style '{style}' (nS={nS}, nB={nB})")
    part = np.array([0]*nS + [1]*nB + [2]*nZ, dtype=int)

    # Latent state dimensions
    ds, db, dz = 2, 4, 2
    d = ds + db + dz
    A_true = _make_true_A(ds, db, dz, sp.rho_dyn, rng)

    # Latent trajectory  x(t) = A x(t-1) + η
    x = np.zeros((T, d))
    for t in range(1, T):
        x[t] = A_true @ x[t-1] + rng.normal(0, 0.03, d)

    Y = np.zeros((T, N, D))

    # ── flat / uninformative data (P3) ───────────────────────────────────────
    if flat_data:
        # All nodes share the same base signal → topology invisible in data
        base = rng.normal(0, 1.0, (T, D))
        for i in range(N):
            Y[:, i, :] = base + rng.normal(0, 0.08 * noise_scale, (T, D))
        return Y, part

    # ── spatially structured data ─────────────────────────────────────────────
    heights = np.linspace(0, 1, N)
    radii   = 0.04 + 0.01 * np.sin(np.pi * heights)

    for i, (h, r) in enumerate(zip(heights, radii)):
        role  = part[i]
        # Two latent dimensions per role block  (S→0:2, B→2:6, Z→6:8)
        offsets = {0: (0, 2), 1: (2, 6), 2: (6, 8)}
        a, b_ = offsets[role]
        xi    = x[:, a : min(b_, d)]
        noise = rng.normal(0, noise_scale, (T, D))

        if   role == 0: T_mean = 22.0
        elif role == 1: T_mean = T_out
        else:           T_mean = sp.T_inner * (1.0 - 0.08 * h)

        c0 = xi[:, 0] if xi.shape[1] > 0 else np.zeros(T)
        c1 = xi[:, 1] if xi.shape[1] > 1 else np.zeros(T)

        Y[:, i, 0] = T_mean + c0 * 3.0   + noise[:, 0] * 0.5     # T_inner
        Y[:, i, 1] = T_out  + c1 * 1.5   + noise[:, 1] * 0.3     # T_outer
        Y[:, i, 2] = (sp.T_inner - T_out) / (sp.wall_mm * 1e-3)  \
                     + noise[:, 2] * 8.0                            # grad_T
        Y[:, i, 3] = 0.04 * (sp.T_inner - T_out) / \
                     (sp.wall_mm * 1e-3) * 1e-3 + noise[:, 3]*1e-4 # flux
        Y[:, i, 4] = r + noise[:, 4] * 1e-3                        # radius
        Y[:, i, 5] = 1.0 / r + noise[:, 5] * 0.1                  # curvature
        Y[:, i, 6] = sp.wall_mm * 1e-3 + noise[:, 6] * 1e-4       # wall_w

    return Y, part
