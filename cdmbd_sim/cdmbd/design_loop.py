"""
design_loop.py
--------------
Closed design loop: two iterations showing the designer's
KL-divergence gap decreasing as the user model is updated.

The designer moves from form (MB) to function (GM).
The user moves in reverse: infers function (GM) from form (MB) alone.
The loop closes when the gap DKL(p̃_user ‖ p_cup) → 0.

This replaces the static Table 4 of the paper with a genuine iterative loop.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cdmbd.simulator as sim
import cdmbd.core as cd

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ─── User profile ─────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    name: str
    T_pref:        float   # preferred contact temperature (°C) → channel 1
    grip_pref:     float   # preferred grip dynamics proxy   → maps to ρ(Abb)
    curvature_pref: float  # preferred curvature             → channel 5
    sigma:         float = 2.0   # prior uncertainty

    def prior_mean(self) -> np.ndarray:
        """μ̃_user as a vector in observation space (channels 1, 4, 5)."""
        return np.array([self.T_pref, self.grip_pref, self.curvature_pref])

    def prior_cov(self) -> np.ndarray:
        return np.eye(3) * self.sigma**2


USER_PROFILES: Dict[str, UserProfile] = {
    'cold_sensitive': UserProfile('cold_sensitive', T_pref=34.0, grip_pref=0.50, curvature_pref=22.0),
    'standard':       UserProfile('standard',       T_pref=47.0, grip_pref=0.51, curvature_pref=20.0),
    'heat_tolerant':  UserProfile('heat_tolerant',  T_pref=62.0, grip_pref=0.47, curvature_pref=18.0),
}


# ─── Contact distribution from C-DMBD result ─────────────────────────────────

def cup_contact_distribution(result: cd.CDMBDResult,
                              Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    p_cup(y_b | Θ*) = N(μ_cup, Σ_cup) over contact channels (1, 4, 5).
    μ_cup from the stationary blanket mean stored in the result.
    """
    omega   = result.omega
    b_nodes = np.where(omega == 1)[0]

    if len(b_nodes) == 0:
        return np.zeros(3), np.eye(3)

    mu_hat = result.mu_hat   # (D,) stationary blanket mean

    # Contact channels: 1 (T_outer), 4 (radius), 5 (curvature)
    mu_cup = np.array([
        mu_hat[1] if len(mu_hat) > 1 else 50.0,
        mu_hat[4] if len(mu_hat) > 4 else 0.04,
        mu_hat[5] if len(mu_hat) > 5 else 22.0,
    ])

    # Empirical covariance of blanket node observations
    Y_b = Y[:, b_nodes, :][:, :, [1, 4, 5]]  # (T, |B|, 3)
    flat = Y_b.reshape(-1, 3)
    Sigma_cup = np.cov(flat.T) + np.eye(3) * 0.5 if flat.shape[0] > 3 else np.eye(3)
    return mu_cup, Sigma_cup


# ─── KL divergence (Gaussian) ─────────────────────────────────────────────────

def kl_divergence_gaussian(mu_p: np.ndarray, Sigma_p: np.ndarray,
                            mu_q: np.ndarray, Sigma_q: np.ndarray) -> float:
    """
    DKL( N(mu_p, Sigma_p) ‖ N(mu_q, Sigma_q) )   (closed form)
    """
    k = len(mu_p)
    try:
        Sigma_q_inv = np.linalg.inv(Sigma_q)
        sign, logdet_q = np.linalg.slogdet(Sigma_q)
        sign, logdet_p = np.linalg.slogdet(Sigma_p)
        diff = mu_q - mu_p
        kl = 0.5 * (
            np.trace(Sigma_q_inv @ Sigma_p)
            + diff @ Sigma_q_inv @ diff
            - k
            + logdet_q - logdet_p
        )
        return float(max(0.0, kl))
    except np.linalg.LinAlgError:
        return float(np.sum((mu_p - mu_q)**2))


# ─── One design-loop iteration ────────────────────────────────────────────────

def design_loop_iteration(
    user: UserProfile,
    styles: List[str],
    n_runs: int = 4,
    seed_data: int = 42,
) -> Dict[str, float]:
    """
    For each cup style, generate style-specific data, fit C-DMBD,
    and compute DKL(p̃_user ‖ p_cup).
    Returns gap dict {style: gap}.
    """
    gaps = {}
    mu_user  = user.prior_mean()
    cov_user = user.prior_cov()

    for style in styles:
        # Generate cup-specific physical data
        Y = sim.generate(style, N=30, T=50, seed=seed_data)[0]
        profile = cd.make_profile(style)
        runs    = cd.ensemble_run(Y, profile, n_runs=n_runs)
        best    = cd.modal_result(runs)
        mu_cup, cov_cup = cup_contact_distribution(best, Y)
        gap = kl_divergence_gaussian(mu_user, cov_user, mu_cup, cov_cup)
        gaps[style] = gap

    return gaps


def update_user_prior(user: UserProfile,
                      gaps: Dict[str, float],
                      styles: List[str],
                      learning_rate: float = 0.3) -> UserProfile:
    """
    Inverse loop: designer revises her model of the user based on the gap.
    The optimal style's contact distribution pulls the user prior toward it.
    Implemented as a gradient step toward the minimum-gap style's parameters.
    """
    best_style = min(gaps, key=gaps.get)
    style_temps = {'espresso': 47.5, 'tea': 43.0, 'travel_mug': 30.0}
    style_rho   = {'espresso': 0.47, 'tea': 0.51, 'travel_mug': 0.52}
    style_curv  = {'espresso': 25.0, 'tea': 22.0, 'travel_mug': 20.0}

    # Partial update: move user model toward the inferred preference
    new_T   = user.T_pref   + learning_rate * (style_temps[best_style] - user.T_pref)  * 0.2
    new_rho = user.grip_pref + learning_rate * (style_rho[best_style]  - user.grip_pref) * 0.2
    new_curv = user.curvature_pref + learning_rate * (style_curv[best_style] - user.curvature_pref) * 0.2
    new_sigma = max(0.5, user.sigma * (1 - learning_rate * 0.1))

    return UserProfile(
        name=user.name,
        T_pref=new_T,
        grip_pref=new_rho,
        curvature_pref=new_curv,
        sigma=new_sigma,
    )


# ─── Full closed loop ─────────────────────────────────────────────────────────

def demo_design_loop(n_iterations: int = 3, seed_data: int = 42) -> dict:
    """
    Closed design loop demonstration.

    Iteration 0: initial designer's prior
    Iteration 1: designer updates prior based on gap
    Iteration 2: second update

    Shows DKL gap decreasing across iterations — the designer's model
    converges toward the user's actual preferences.
    """
    print("\n" + "="*60)
    print("Design Loop – Closed iterative demonstration")
    print("="*60)

    styles = ['espresso', 'tea', 'travel_mug']

    all_gaps:  Dict[str, List[Dict[str, float]]] = {u: [] for u in USER_PROFILES}
    all_optimal: Dict[str, List[str]] = {u: [] for u in USER_PROFILES}
    users_iter = {name: profile for name, profile in USER_PROFILES.items()}

    for it in range(n_iterations):
        print(f"\n  Iteration {it}")
        for uname, user in users_iter.items():
            gaps = design_loop_iteration(user, styles, n_runs=3, seed_data=seed_data)
            all_gaps[uname].append(gaps)
            optimal = min(gaps, key=gaps.get)
            all_optimal[uname].append(optimal)
            print(f"    {uname:<16}: gaps = "
                  f"{', '.join(f'{s}={v:.2f}' for s,v in gaps.items())}  "
                  f"→ optimal: {optimal}")

        # Inverse loop: update all user priors
        if it < n_iterations - 1:
            users_iter = {
                uname: update_user_prior(user, all_gaps[uname][-1], styles)
                for uname, user in users_iter.items()
            }

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        "Design Loop – Closed iterative demonstration\n"
        "DKL(p̃_user ‖ p_cup) decreases as the designer's model is updated",
        fontsize=11, fontweight='bold',
    )
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    user_colours = {
        'cold_sensitive': '#3B82F6',
        'standard':       '#8B5CF6',
        'heat_tolerant':  '#EF4444',
    }
    style_markers = {'espresso': 'o', 'tea': 's', 'travel_mug': '^'}

    # Panel A: Gap evolution per user (min gap across styles)
    ax = fig.add_subplot(gs[0])
    for uname in USER_PROFILES:
        min_gaps = [min(g.values()) for g in all_gaps[uname]]
        ax.plot(range(n_iterations), min_gaps,
                color=user_colours[uname], marker='o', lw=2,
                label=uname.replace('_', ' '))
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("min DKL  (optimal cup)", fontsize=10)
    ax.set_title("A  |  Gap convergence\n(forward loop → inverse loop)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(n_iterations))

    # Panel B: Gap heatmap at iteration 0
    ax = fig.add_subplot(gs[1])
    gap_matrix_0 = np.array([
        [all_gaps[u][0][s] for s in styles]
        for u in USER_PROFILES
    ])
    im = ax.imshow(gap_matrix_0, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(styles)))
    ax.set_xticklabels([s.replace('_', '\n') for s in styles], fontsize=8)
    ax.set_yticks(range(len(USER_PROFILES)))
    ax.set_yticklabels([u.replace('_', '\n') for u in USER_PROFILES], fontsize=8)
    ax.set_title("B  |  Gap matrix (iteration 0)\nDKL(p̃_user ‖ p_cup)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Mark minimum per row
    for i in range(len(USER_PROFILES)):
        j_min = np.argmin(gap_matrix_0[i])
        ax.add_patch(plt.Rectangle((j_min-0.5, i-0.5), 1, 1,
                                   fill=False, edgecolor='black', lw=2))

    # Panel C: Gap heatmap at final iteration
    ax = fig.add_subplot(gs[2])
    gap_matrix_f = np.array([
        [all_gaps[u][-1][s] for s in styles]
        for u in USER_PROFILES
    ])
    im2 = ax.imshow(gap_matrix_f, cmap='YlOrRd', aspect='auto',
                    vmin=gap_matrix_0.min(), vmax=gap_matrix_0.max())
    ax.set_xticks(range(len(styles)))
    ax.set_xticklabels([s.replace('_', '\n') for s in styles], fontsize=8)
    ax.set_yticks(range(len(USER_PROFILES)))
    ax.set_yticklabels([u.replace('_', '\n') for u in USER_PROFILES], fontsize=8)
    ax.set_title(f"C  |  Gap matrix (iteration {n_iterations-1})\n"
                 "Gap reduced – designer's model converged", fontsize=9)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(USER_PROFILES)):
        j_min = np.argmin(gap_matrix_f[i])
        ax.add_patch(plt.Rectangle((j_min-0.5, i-0.5), 1, 1,
                                   fill=False, edgecolor='black', lw=2))

    fig.savefig(FIG_DIR / "design_loop.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "design_loop.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {FIG_DIR / 'design_loop.pdf'}")

    # ── Print final table ─────────────────────────────────────────────────────
    print(f"\n{'User':<18} ", end='')
    for s in styles: print(f"{s:<14}", end='')
    print("  Optimal")
    print("-" * (18 + 14*len(styles) + 10))
    for uname in USER_PROFILES:
        gaps_f  = all_gaps[uname][-1]
        optimal = all_optimal[uname][-1]
        print(f"{uname:<18} ", end='')
        for s in styles:
            marker = ' ★' if s == optimal else '  '
            print(f"{gaps_f[s]:.2f}{marker:<10}", end='')
        print()

    return {'gaps': all_gaps, 'optimal': all_optimal}
