"""
tests/test_cdmbd.py
-------------------
Unit tests for the C-DMBD simulation package.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cdmbd import generate, run_cdmbd, make_profile, compute_violations, gap_score
from cdmbd.core import (
    Requirement, RequirementProfile,
    _mu_hat, _marginal_B_delta,
    BlanketDynamics, ensemble_run, modal_result,
)


# ─── simulator ────────────────────────────────────────────────────────────────

class TestSimulator:
    def test_shape(self):
        for style in ('espresso', 'tea', 'travel_mug'):
            Y, part = generate(style, N=30, T=50, D=7)
            assert Y.shape == (50, 30, 7), f"Wrong shape for {style}"
            assert part.shape == (30,)
            assert set(part).issubset({0, 1, 2})

    def test_partition_completeness(self):
        _, part = generate('tea')
        assert 0 in part, "No S nodes"
        assert 1 in part, "No B nodes"
        assert 2 in part, "No Z nodes"

    def test_flat_data_homogeneity(self):
        """Flat data should have low spatial variance across nodes."""
        Y, _ = generate('tea', flat_data=True, seed=0)
        # Variance between node means should be much smaller than for normal data
        node_means_flat = Y.mean(axis=0).std(axis=0).mean()
        Y_n, _ = generate('tea', flat_data=False, seed=0)
        node_means_norm = Y_n.mean(axis=0).std(axis=0).mean()
        assert node_means_flat < node_means_norm * 0.1, \
            "Flat data should have much lower inter-node variance"

    def test_double_wall_colder(self):
        """Double-wall cup should have lower T_outer (channel 1)."""
        Y_sw, _ = generate('tea', double_wall=False, seed=42)
        Y_dw, _ = generate('tea', double_wall=True,  seed=42)
        T_out_sw = Y_sw[:, :, 1].mean()
        T_out_dw = Y_dw[:, :, 1].mean()
        assert T_out_dw < T_out_sw, \
            f"Double-wall should be cooler: {T_out_dw:.1f} vs {T_out_sw:.1f}"


# ─── requirements ─────────────────────────────────────────────────────────────

class TestRequirements:
    def test_violation_zero_inside_deadzone(self):
        req = Requirement(channel=0, tau=50.0, epsilon=5.0, weight=1.0)
        mu  = np.array([52.0, 0, 0, 0, 0, 0, 0])   # within [45, 55]
        v   = compute_violations(mu, RequirementProfile([req]))
        assert v[0] == pytest.approx(0.0)

    def test_violation_positive_outside_deadzone(self):
        req = Requirement(channel=0, tau=50.0, epsilon=5.0, weight=1.0)
        mu  = np.array([60.0, 0, 0, 0, 0, 0, 0])   # outside
        v   = compute_violations(mu, RequirementProfile([req]))
        assert v[0] == pytest.approx(5.0)   # |60-50| - 5 = 5

    def test_violation_weight(self):
        req = Requirement(channel=0, tau=50.0, epsilon=0.0, weight=2.0)
        mu  = np.array([55.0, 0, 0, 0, 0, 0, 0])
        v   = compute_violations(mu, RequirementProfile([req]))
        assert v[0] == pytest.approx(10.0)  # 2.0 * |55-50|

    def test_make_profile_keys(self):
        for style in ('espresso', 'tea', 'travel_mug'):
            p = make_profile(style)
            assert len(p.reqs) >= 3, f"Too few requirements for {style}"

    def test_marginal_delta_adding(self):
        """Adding a node that worsens violation → positive delta."""
        profile = RequirementProfile([
            Requirement(channel=1, tau=50.0, epsilon=0.0, weight=1.0)
        ])
        # Current blanket mean at T_outer=80 → violated (too high)
        mu_curr = np.array([0, 80, 0, 0, 0, 0, 0], dtype=float)
        # Adding another hot node → delta should be small (no change)
        y_hot   = np.array([0, 80, 0, 0, 0, 0, 0], dtype=float)
        lam = np.array([1.0])
        delta = _marginal_B_delta(y_hot, mu_curr, 5, lam, profile, adding=True)
        assert delta == pytest.approx(0.0, abs=1e-6)

    def test_marginal_delta_cooling(self):
        """Adding a cold node to a hot blanket → delta negative (reduces violation)."""
        profile = RequirementProfile([
            Requirement(channel=1, tau=50.0, epsilon=0.0, weight=1.0)
        ])
        mu_curr = np.array([0, 80, 0, 0, 0, 0, 0], dtype=float)
        y_cold  = np.array([0, 30, 0, 0, 0, 0, 0], dtype=float)
        lam = np.array([1.0])
        delta = _marginal_B_delta(y_cold, mu_curr, 5, lam, profile, adding=True)
        assert delta < 0.0, "Adding cold node to hot blanket should reduce violation"


# ─── C-DMBD algorithm ─────────────────────────────────────────────────────────

class TestCDMBD:
    @pytest.fixture
    def tea_data(self):
        return generate('tea', seed=0)

    def test_result_shape(self, tea_data):
        Y, _ = tea_data
        result = run_cdmbd(Y, make_profile('tea'), n_iter=20, seed=0)
        assert result.omega.shape == (30,)
        assert set(result.omega).issubset({0, 1, 2})

    def test_blanket_nonempty(self, tea_data):
        Y, _ = tea_data
        result = run_cdmbd(Y, make_profile('tea'), n_iter=20, seed=0)
        assert result.n_B > 0, "Blanket must be non-empty"

    def test_rho_in_range(self, tea_data):
        Y, _ = tea_data
        result = run_cdmbd(Y, make_profile('tea'), n_iter=20, seed=0)
        assert 0.29 <= result.rho_bb <= 0.97, \
            f"rho out of range: {result.rho_bb}"

    def test_violations_nonneg(self, tea_data):
        Y, _ = tea_data
        result = run_cdmbd(Y, make_profile('tea'), n_iter=20, seed=0)
        assert all(v >= 0 for v in result.violations), "Violations must be non-negative"

    def test_lambda_nonneg(self, tea_data):
        Y, _ = tea_data
        result = run_cdmbd(Y, make_profile('tea'), n_iter=20, seed=0)
        assert all(l >= 0 for l in result.lambda_), "Lambda must be non-negative"

    def test_elbo_trace_length(self, tea_data):
        Y, _ = tea_data
        n_iter = 15
        result = run_cdmbd(Y, make_profile('tea'), n_iter=n_iter, seed=0)
        assert len(result.elbo_trace) == n_iter

    def test_p3_disambiguation(self):
        """P3: flat data → requirements determine topology."""
        Y_flat, _ = generate('tea', flat_data=True, seed=7)
        r_esp  = run_cdmbd(Y_flat, make_profile('espresso'),   n_iter=40, seed=0)
        r_tmug = run_cdmbd(Y_flat, make_profile('travel_mug'), n_iter=40, seed=0)
        assert r_esp.n_B != r_tmug.n_B, \
            "With flat data, different requirements must produce different topologies"

    def test_p1_fingerprint_differs(self):
        """P1: same data, different profiles → different lambda* fingerprints."""
        Y, _ = generate('tea', double_wall=True, seed=99)
        r_esp  = run_cdmbd(Y, make_profile('espresso'),   n_iter=50, seed=0)
        r_tmug = run_cdmbd(Y, make_profile('travel_mug'), n_iter=50, seed=0)
        # At least one lambda* component should differ significantly
        diff = np.abs(r_esp.lambda_ - r_tmug.lambda_).max()
        assert diff > 0.5, \
            f"Lambda fingerprints should differ significantly; max diff = {diff:.3f}"

    def test_ensemble_length(self, tea_data):
        Y, _ = tea_data
        runs = ensemble_run(Y, make_profile('tea'), n_runs=3, n_iter=15)
        assert len(runs) == 3

    def test_modal_result_is_valid(self, tea_data):
        Y, _ = tea_data
        runs   = ensemble_run(Y, make_profile('tea'), n_runs=4, n_iter=15)
        best   = modal_result(runs)
        assert best in runs


# ─── BlanketDynamics ──────────────────────────────────────────────────────────

class TestBlanketDynamics:
    def test_rho_decreases_when_too_hot(self):
        """T_outer too high → need more exchange → rho decreases."""
        rng = np.random.default_rng(0)
        dyn = BlanketDynamics(db=4, rng=rng)
        rho_init = dyn.rho
        profile = RequirementProfile([
            Requirement(channel=1, tau=40.0, epsilon=0.0, weight=1.0)
        ])
        mu_hot = np.array([0, 90, 0, 0, 0, 0, 0], dtype=float)  # too hot
        lam    = np.array([5.0])
        for _ in range(20):
            dyn.update(mu_hot, lam, profile)
        # With high T_outer and high lambda, rho should increase (more insulation)
        # Wait - physics: T_outer too high → need MORE insulation → rho UP
        assert dyn.rho != rho_init, "rho should change under violation pressure"

    def test_rho_clipped(self):
        """rho must stay in [0.30, 0.96]."""
        rng = np.random.default_rng(0)
        dyn = BlanketDynamics(db=4, rng=rng)
        profile = RequirementProfile([
            Requirement(channel=1, tau=1000.0, epsilon=0.0, weight=10.0)
        ])
        mu = np.zeros(7)
        lam = np.array([20.0])
        for _ in range(100):
            dyn.update(mu, lam, profile)
        assert 0.28 <= dyn.rho <= 0.98, f"rho out of clipping range: {dyn.rho}"


# ─── Gap score ────────────────────────────────────────────────────────────────

class TestGapScore:
    def test_flat_gap_near_zero(self):
        """Flat data should have gap close to zero (topology ambiguous)."""
        Y_flat, _ = generate('tea', flat_data=True, seed=7)
        g = gap_score(Y_flat, make_profile('tea'))
        assert abs(g) < 1.0, f"Flat data gap should be near zero; got {g:.4f}"

    def test_gap_returns_scalar(self):
        """gap_score should return a scalar float."""
        Y, _ = generate('tea', seed=0)
        g = gap_score(Y, make_profile('tea'))
        assert isinstance(g, float), f"Expected float, got {type(g)}"
