"""
phenomena.py
------------
Computational demonstrations of the three phenomena identified in the paper:

    P1 – Intra-family navigation
    P2 – Family transition (phase transition of interface type)
    P3 – Ontological disambiguation
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Tuple

import cdmbd.simulator as sim
import cdmbd.core as cd


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# P1  –  Intra-family navigation
# ─────────────────────────────────────────────────────────────────────────────

def demo_p1(n_runs: int = 6, seed_data: int = 99) -> dict:
    """
    P1: Intra-family navigation.

    Key setup (matching Table 3 of the paper):
      – double-wall data: T_outer ≈ 31°C
      – travel_mug requirement: T_outer target ∈ [22, 38] → SATISFIED → λ*_R1 = 0
      – espresso  requirement: T_outer target ∈ [40, 55] → VIOLATED  → λ*_R1 > 0

    Same physical substrate; different requirement profiles.
    Expected: different rho(A_bb) and different λ* fingerprint.
    (Topology may also differ — showing both navigation and transition.)
    """
    print("\n" + "="*60)
    print("P1 – Intra-family navigation")
    print("="*60)

    # Double-wall data: T_outer ≈ 31°C
    Y, _ = sim.generate('tea', N=30, T=50, double_wall=True, seed=seed_data)

    styles_p1 = ['espresso', 'travel_mug']
    results = {}
    for style in styles_p1:
        profile = cd.make_profile(style)
        runs = cd.ensemble_run(Y, profile, n_runs=n_runs,
                               alpha_lambda=0.10, lambda_max=15.0, n_iter=55)
        results[style] = {
            'runs':         runs,
            'best':         cd.modal_result(runs),
            'nB_modal':     cd.modal_result(runs).n_B,
            'rho_all':      [r.rho_bb for r in runs],
            'lam_norm_all': [r.lambda_norm for r in runs],
            'lam_best':     cd.modal_result(runs).lambda_,
        }

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\n{'Style':<14} {'|B| modal':<12} {'ρ(A_bb)':<22} {'‖λ*‖₁'}")
    print("-" * 62)
    for s, r in results.items():
        rho_m = np.mean(r['rho_all']); rho_s = np.std(r['rho_all'])
        lam_m = np.mean(r['lam_norm_all'])
        print(f"{s:<14} {r['nB_modal']:<12} "
              f"{rho_m:.3f} ± {rho_s:.3f}        {lam_m:.2f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("P1 – Intra-family navigation\n"
                 "Same physical data (double-wall, T_outer≈31°C) · "
                 "different requirements → different mode",
                 fontsize=11, fontweight='bold')

    colours = {'espresso': '#F59E0B', 'travel_mug': '#10B981'}
    labels  = {'espresso': 'espresso\n(T_outer target=47.5°C\n→ VIOLATED)', 
               'travel_mug': 'travel_mug\n(T_outer target=30°C\n→ SATISFIED)'}

    # Panel A: ρ(A_bb) — dynamical mode
    ax = axes[0]
    for j, (s, r) in enumerate(results.items()):
        rhos = r['rho_all']
        jit  = np.random.default_rng(j).normal(0, 0.04, len(rhos))
        ax.scatter(np.full(len(rhos), j) + jit, rhos,
                   color=colours[s], s=80, zorder=3, alpha=0.85)
        ax.plot([j-0.25, j+0.25], [np.mean(rhos)]*2, color=colours[s], lw=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([labels[s] for s in styles_p1], fontsize=8)
    ax.set_ylabel("ρ(A_bb) – spectral radius", fontsize=10)
    ax.set_title("A  |  Dynamical mode\n(ρ encodes speed of heat exchange)", fontsize=9)
    ax.set_ylim(0.25, 1.02)
    ax.axhspan(0.25, 0.50, alpha=0.06, color='blue',  label='fast exchange zone')
    ax.axhspan(0.80, 1.02, alpha=0.06, color='red',   label='insulation zone')
    ax.legend(fontsize=7, loc='center right')
    ax.grid(axis='y', alpha=0.3)

    # Panel B: |B| — topology
    ax = axes[1]
    for j, (s, r) in enumerate(results.items()):
        nBs = [run.n_B for run in r['runs']]
        jit = np.random.default_rng(j+5).normal(0, 0.04, len(nBs))
        ax.scatter(np.full(len(nBs), j) + jit, nBs,
                   color=colours[s], s=80, zorder=3, alpha=0.85)
        ax.plot([j-0.25, j+0.25], [np.mean(nBs)]*2, color=colours[s], lw=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([s.replace('_', '\n') for s in styles_p1], fontsize=9)
    ax.set_ylabel("|B| – blanket node count", fontsize=10)
    ax.set_title("B  |  Topology\n(may differ → P2 also in play)", fontsize=9)
    ax.set_ylim(0, 25)
    ax.grid(axis='y', alpha=0.3)

    # Panel C: λ* per-requirement fingerprint
    ax = axes[2]
    req_names = [r.name for r in cd.make_profile('espresso').reqs]
    x = np.arange(len(req_names))
    w = 0.35
    for j, (s, r) in enumerate(results.items()):
        ax.bar(x + j*w, r['lam_best'], w, label=s.replace('_', ' '),
               color=colours[s], alpha=0.88)
    ax.set_xticks(x + w/2)
    ax.set_xticklabels(req_names, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel("λ*_k (Lagrange multiplier at convergence)", fontsize=9)
    ax.set_title("C  |  Ontological fingerprint\n"
                 "R1 binding for espresso only (VIOLATED)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P1_intra_family_navigation.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "P1_intra_family_navigation.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {FIG_DIR / 'P1_intra_family_navigation.pdf'}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# P2  –  Family transition
# ─────────────────────────────────────────────────────────────────────────────

def demo_p2(n_temps: int = 20, seed_data: int = 42) -> dict:
    """
    Scan the outer-temperature requirement R1 from 25°C to 80°C.
    Expected: discontinuous jump in |B| at a critical temperature.

    Uses a cup with gap Δ ≈ 0 (flat data) so that small requirement
    changes can flip the topology.
    """
    print("\n" + "="*60)
    print("P2 – Family transition (phase transition of interface type)")
    print("="*60)

    # Low-gap data: use flat=True for spatial ambiguity
    Y, _ = sim.generate('tea', N=30, T=50, flat_data=False,
                        noise_scale=3.0, seed=seed_data)

    temps   = np.linspace(25.0, 80.0, n_temps)
    nBs     = []
    gaps    = []

    # Baseline gap (no requirements)
    base_profile = cd.RequirementProfile([
        cd.Requirement(channel=1, tau=50.0, epsilon=100.0, weight=0.0, name='dummy')
    ])
    gap_val = cd.gap_score(Y, base_profile, seed=0)
    print(f"\nData gap Δ(ω*) ≈ {gap_val:.3f}  (predicted topological instability if ≈ 0)")

    for i, T_target in enumerate(temps):
        profile = cd.RequirementProfile([
            cd.Requirement(channel=1, tau=T_target, epsilon=5.0, weight=1.0, name='R1_temp'),
            cd.Requirement(channel=4, tau=0.045,    epsilon=0.01, weight=0.3, name='R2_rad'),
        ])
        runs = [cd.run_cdmbd(Y, profile, n_iter=35,
                             alpha_lambda=0.10, seed=s) for s in range(4)]
        nBs_run = [r.n_B for r in runs]
        modal_nB = max(set(nBs_run), key=nBs_run.count)
        nBs.append(modal_nB)
        gaps.append(gap_val)
        print(f"  T_target={T_target:.1f}°C → |B|={modal_nB}", end='\r')

    print()

    # Detect transition point
    transitions = [i for i in range(1, len(nBs)) if nBs[i] != nBs[i-1]]
    print(f"\nTopology jumps detected at T_target ≈ "
          f"{[f'{temps[i]:.1f}°C' for i in transitions]}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("P2 – Family transition\n"
                 "Continuous requirement scan → discontinuous topology change",
                 fontsize=12, fontweight='bold')

    # Panel A: |B| vs T_target
    ax = axes[0]
    ax.step(temps, nBs, where='post', color='#EF4444', lw=2.5, label='|B| (modal)')
    for t_idx in transitions:
        ax.axvline(temps[t_idx], color='grey', ls=':', lw=1.5, alpha=0.7)
        ax.text(temps[t_idx]+0.5, max(nBs)+0.5, f'jump\n≈{temps[t_idx]:.0f}°C',
                fontsize=7, color='grey')
    ax.set_xlabel("R1 target temperature (°C)", fontsize=10)
    ax.set_ylabel("|B| – blanket node count", fontsize=10)
    ax.set_title("A  |  Topology as function of requirement", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Panel B: data gap annotation
    ax = axes[1]
    ax.axhline(gap_val, color='#6366F1', lw=2.5, label=f'Δ(ω*) = {gap_val:.3f}')
    ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
    ax.fill_between([0, 1], [0, 0], [gap_val, gap_val],
                    color='#6366F1', alpha=0.15,
                    label='topological instability zone')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, gap_val * 3)
    ax.set_xticks([])
    ax.set_ylabel("Data gap Δ(ω*)", fontsize=10)
    ax.set_title("B  |  Gap score\n(near-zero gap → transition predicted)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, gap_val * 1.3,
            "Gap computed from\nrequirement-free inference\n(independent prediction)",
            ha='center', fontsize=8, color='#6366F1',
            bbox=dict(boxstyle='round', fc='white', ec='#6366F1', alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P2_family_transition.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "P2_family_transition.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Figure saved → {FIG_DIR / 'P2_family_transition.pdf'}")

    return {'temps': temps, 'nBs': nBs, 'gap': gap_val, 'transitions': transitions}


# ─────────────────────────────────────────────────────────────────────────────
# P3  –  Ontological disambiguation
# ─────────────────────────────────────────────────────────────────────────────

def demo_p3(n_runs: int = 5, seed_data: int = 7) -> dict:
    """
    Spatially flat data (low gap ≈ 0): data alone cannot determine topology.
    Espresso vs travel_mug requirements → different |B|.

    This demonstrates Proposition 3b: requirements are the dominant force
    shaping the interface type when the data are weakly informative.
    """
    print("\n" + "="*60)
    print("P3 – Ontological disambiguation")
    print("="*60)

    # Flat data: same signal for all nodes → gap ≈ 0
    Y_flat, _ = sim.generate('tea', N=30, T=50, flat_data=True, seed=seed_data)

    # Measure gap
    base_profile = cd.RequirementProfile([
        cd.Requirement(channel=1, tau=50.0, epsilon=100.0, weight=0.0, name='dummy')
    ])
    gap_flat = cd.gap_score(Y_flat, base_profile, seed=0)
    print(f"\nData gap on flat data: Δ(ω*) ≈ {gap_flat:.4f}  (ideally ≈ 0)")

    results = {}
    for style in ('espresso', 'travel_mug'):
        profile = cd.make_profile(style)
        runs    = [cd.run_cdmbd(Y_flat, profile, n_iter=40,
                                alpha_lambda=0.12, seed=s) for s in range(n_runs)]
        nBs     = [r.n_B for r in runs]
        modal_nB = max(set(nBs), key=nBs.count)
        best     = cd.modal_result(runs)
        results[style] = {
            'runs': runs,
            'best': best,
            'nB_modal': modal_nB,
            'nBs': nBs,
        }
        print(f"  {style:<14}: |B| counts = {sorted(nBs)}  → modal |B| = {modal_nB}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("P3 – Ontological disambiguation\n"
                 "Flat data (gap ≈ 0): requirements determine the interface type",
                 fontsize=12, fontweight='bold')

    colours = {'espresso': '#F59E0B', 'travel_mug': '#10B981'}

    # Panel A: |B| distribution for each style
    ax = axes[0]
    for j, (style, r) in enumerate(results.items()):
        nBs = r['nBs']
        jitter = np.random.default_rng(j).normal(0, 0.05, len(nBs))
        ax.scatter(np.full(len(nBs), j) + jitter, nBs,
                   color=colours[style], s=80, zorder=3, alpha=0.8,
                   label=style.replace('_', ' '))
        ax.axhline(r['nB_modal'], color=colours[style], ls='--', lw=1.5, alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['espresso', 'travel_mug'], fontsize=10)
    ax.set_ylabel("|B| – blanket size", fontsize=10)
    ax.set_title(f"A  |  Topology (gap Δ ≈ {gap_flat:.3f})\n"
                 "Requirements resolve topological ambiguity", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: partition visualisation for best run of each style
    ax = axes[1]
    N = 30
    role_colours = {0: '#CBD5E1', 1: '#FBBF24', 2: '#60A5FA'}
    role_labels  = {0: 'S (env)', 1: 'B (blanket)', 2: 'Z (interior)'}
    offsets      = {'espresso': -0.2, 'travel_mug': 0.2}

    for style, r in results.items():
        omega = r['best'].omega
        for i in range(N):
            role = omega[i]
            ax.barh(i, 1, left=offsets[style]*2, height=0.8,
                    color=role_colours[role], alpha=0.75)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=role_colours[k], label=role_labels[k])
                      for k in [0, 1, 2]]
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right')
    ax.set_xlabel("← espresso   |   travel_mug →", fontsize=9)
    ax.set_ylabel("Node index (0 = bottom, 29 = top)", fontsize=9)
    ax.set_title("B  |  Inferred partitions from same flat data\n"
                 "Different requirements → different topology", fontsize=9)
    ax.set_xlim(-0.8, 0.8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P3_ontological_disambiguation.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "P3_ontological_disambiguation.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Figure saved → {FIG_DIR / 'P3_ontological_disambiguation.pdf'}")

    return results
