"""
cdmbd.py  (v2)
--------------
Constrained Dynamic Markov Blanket Detection (C-DMBD).

Key architectural fix: the E-step computes the MARGINAL effect of assigning
each node to B on the requirement violations. This is the mechanism that makes
requirements change the partition topology (P2, P3) and navigate within it (P1).

Mathematical core
─────────────────
Blanket mean (Eq. 4, approximated via empirical role means):
    mu_hat_b ≈ (1/|B|) Σ_{i∈B} y_bar_i     (per channel)

Violation (Eq. 5):
    v_k = w_k max(0, |phi_k − tau_k| − eps_k),   phi_k = mu_hat_b[channel_k]

Marginal penalty for assigning node i to B (Algorithm 1, E-step):
    delta(i, B) = Σ_k lambda_k * [v_k(B ∪ {i}) − v_k(B)]

Augmented node score:
    score(i, ell) = -||y_i − mu_ell||² / (2σ²)  −  lambda_sum * delta(i, ell)

M-step: update role means; update rho(A_bb) via violation gradient (Eq. 8).
Dual (Eq. 11): lambda_k ← clip(lambda_k + alpha * v_k, 0, lambda_max).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# ─── Requirement definitions ──────────────────────────────────────────────────

@dataclass
class Requirement:
    """
    Single functional requirement on the blanket stationary statistics.
    phi_k = mu_hat_b[channel]
    v_k   = weight * max(0, |phi_k − tau| − epsilon)
    """
    channel: int
    tau:     float
    epsilon: float
    weight:  float = 1.0
    name:    str   = "R"


@dataclass
class RequirementProfile:
    reqs: List[Requirement]
    def __iter__(self): return iter(self.reqs)
    def __len__(self):  return len(self.reqs)


def make_profile(style: str) -> RequirementProfile:
    """Standard requirement profile for a cup style."""
    # channels: 0=T_inner, 1=T_outer, 2=grad_T, 3=flux, 4=radius, 5=curvature, 6=wall_w
    _p = {
        'espresso': [
            Requirement(1, 47.5,  7.5,  1.5, 'R1_Touter'),
            Requirement(0, 88.0,  4.0,  1.0, 'R2_Tinner'),
            Requirement(6, 0.003, 0.001,0.8, 'R3_wall'),
            Requirement(4, 0.035, 0.007,0.5, 'R4_radius'),
        ],
        'tea': [
            Requirement(1, 43.0,  7.0,  1.5, 'R1_Touter'),
            Requirement(0, 84.0,  4.0,  1.0, 'R2_Tinner'),
            Requirement(6, 0.004, 0.001,0.8, 'R3_wall'),
            Requirement(4, 0.045, 0.008,0.5, 'R4_radius'),
        ],
        'travel_mug': [
            Requirement(1, 30.0,  8.0,  1.5, 'R1_Touter'),
            Requirement(0, 80.0,  4.0,  1.0, 'R2_Tinner'),
            Requirement(6, 0.008, 0.002,0.8, 'R3_wall'),
            Requirement(4, 0.050, 0.008,0.5, 'R4_radius'),
        ],
    }
    return RequirementProfile(_p[style])


# ─── Violation functions ──────────────────────────────────────────────────────

def _mu_hat(Y_mean: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Approximate blanket stationary mean as empirical mean of B-node observations.
    Y_mean: (N, D).  Returns (D,).
    """
    mask = omega == 1
    return Y_mean[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(Y_mean.shape[1])


def compute_violations(mu_hat: np.ndarray,
                       profile: RequirementProfile) -> np.ndarray:
    """v_k = w_k max(0, |phi_k − tau_k| − eps_k)  (Eq. 5)"""
    v = np.zeros(len(profile.reqs))
    for k, req in enumerate(profile.reqs):
        phi = mu_hat[req.channel] if req.channel < len(mu_hat) else 0.0
        v[k] = req.weight * max(0.0, abs(phi - req.tau) - req.epsilon)
    return v


def _marginal_B_delta(y_i: np.ndarray,
                      mu_hat_curr: np.ndarray,
                      n_B: int,
                      lambda_: np.ndarray,
                      profile: RequirementProfile,
                      adding: bool) -> float:
    """
    Change in Σ_k lambda_k v_k when node with mean y_i is added to
    (adding=True) or removed from (adding=False) B.
    """
    n_new = n_B + (1 if adding else -1)
    if n_new <= 0:
        return 0.0
    if adding:
        mu_new = (mu_hat_curr * n_B + y_i) / n_new
    else:
        mu_new = ((mu_hat_curr * n_B - y_i) / n_new
                  if n_B > 1 else np.zeros_like(mu_hat_curr))
    v_old = compute_violations(mu_hat_curr, profile)
    v_new = compute_violations(mu_new,      profile)
    return float(np.dot(lambda_, v_new - v_old))


# ─── A_bb spectral radius model ───────────────────────────────────────────────

class BlanketDynamics:
    """
    Tracks the spectral radius rho of A_bb.
    Gradient: insulation (rho↑) reduces heat exchange (lowers T_outer),
              rho↓ increases exchange (raises T_outer).
    """
    def __init__(self, db: int, rng: np.random.Generator):
        self.db  = db
        self.rho = float(rng.uniform(0.45, 0.55))

    @property
    def A_bb(self) -> np.ndarray:
        return self.rho * np.eye(self.db)

    def update(self, mu_hat: np.ndarray, lambda_: np.ndarray,
               profile: RequirementProfile, lr: float = 0.025):
        """
        Gradient step on rho via violation (Eq. 8, scalar version).

        Physics: rho↑ = more insulation = lower T_outer
                 rho↓ = more exchange   = higher T_outer

        Gradient of v_k w.r.t. rho:
          - phi > tau (too hot):  want phi↓ → want rho↑ → positive gradient
          - phi < tau (too cold): want phi↑ → want rho↓ → negative gradient
        """
        grad = 0.0
        for k, req in enumerate(profile.reqs):
            if lambda_[k] < 1e-8:
                continue
            phi  = mu_hat[req.channel] if req.channel < len(mu_hat) else 0.0
            diff = phi - req.tau
            if abs(diff) <= req.epsilon:
                continue
            sign = np.sign(diff)
            if req.channel == 1:   # T_outer: dominant channel
                grad += lambda_[k] * req.weight * sign * 0.18
            elif req.channel == 0: # T_inner
                grad += lambda_[k] * req.weight * sign * 0.04
        self.rho = float(np.clip(self.rho + lr * grad, 0.30, 0.96))


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class CDMBDResult:
    omega:      np.ndarray
    rho_bb:     float
    lambda_:    np.ndarray
    violations: np.ndarray
    mu_hat:     np.ndarray
    elbo_trace: List[float]
    n_B: int

    @property
    def lambda_norm(self) -> float:
        return float(np.sum(self.lambda_))


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_cdmbd(
    Y:            np.ndarray,
    profile:      RequirementProfile,
    db:           int   = 4,
    n_iter:       int   = 50,
    alpha_lambda: float = 0.12,
    lambda_max:   float = 20.0,
    n_S_fixed:    int   = 3,
    seed:         int   = 0,
) -> CDMBDResult:
    """
    One C-DMBD chain.

    E-step: for each node i compute
        score(i, ell) = -||y_i − mu_ell||² / (2σ²) − penalty(i, ell)
        where penalty(i, B) = lambda_sum * delta_B(i)  (marginal violation change)
    M-step: update role means; update rho(A_bb) via violation gradient.
    Dual:   lambda_k ← clip(lambda_k + alpha * v_k, 0, lambda_max).
    """
    T_, N, D = Y.shape
    rng       = np.random.default_rng(seed)
    K         = len(profile.reqs)
    Y_mean    = Y.mean(axis=0)           # (N, D) empirical node means

    # ── Initialise ────────────────────────────────────────────────────────────
    s_fixed = list(range(n_S_fixed))
    omega   = np.zeros(N, dtype=int)
    omega[s_fixed] = 0
    split   = n_S_fixed + max(3, (N - n_S_fixed) // 4)
    omega[n_S_fixed:split] = 1
    omega[split:]          = 2

    mu_role: Dict[int, np.ndarray] = {
        r: Y_mean[omega == r].mean(axis=0)
        if (omega == r).sum() > 0 else Y_mean.mean(axis=0)
        for r in range(3)
    }

    lambda_  = np.zeros(K)
    dynamics = BlanketDynamics(db, rng)
    sigma_sq = max(float(np.var(Y_mean)), 1e-2)
    elbo_trace: List[float] = []

    for it in range(n_iter):
        mu_hat   = _mu_hat(Y_mean, omega)
        viols    = compute_violations(mu_hat, profile)
        n_B_curr = int((omega == 1).sum())
        lam_sum  = float(np.sum(lambda_)) + 1e-9

        # ── E-step ────────────────────────────────────────────────────────────
        new_omega = omega.copy()
        for i in range(N):
            if i in s_fixed:
                new_omega[i] = 0
                continue

            y_i    = Y_mean[i]
            scores = np.full(3, -1e12)

            for role in (1, 2):   # only B and Z compete (S is fixed at boundary)
                mu_r   = mu_role[role]
                ll     = -np.sum((y_i - mu_r)**2) / (2 * sigma_sq)
                if role == 1:
                    # Marginal requirement penalty of assigning i to B
                    adding = (omega[i] != 1)
                    delta  = _marginal_B_delta(y_i, mu_hat, n_B_curr,
                                               lambda_, profile, adding=True)
                    # Penalty: positive delta = violation increases = bad
                    penalty = lam_sum * max(0.0, delta) * 5.0
                    scores[role] = ll - penalty
                else:
                    scores[role] = ll

            new_omega[i] = int(np.argmax(scores))

        # Guard: ensure all roles populated
        if (new_omega == 1).sum() < 2:
            pivot = n_S_fixed + (N - n_S_fixed) // 4
            new_omega[pivot:pivot+3] = 1
        if (new_omega == 2).sum() == 0:
            new_omega[-2:] = 2

        omega = new_omega

        # ── M-step ────────────────────────────────────────────────────────────
        for r in range(3):
            mask = omega == r
            if mask.sum() > 0:
                mu_role[r] = Y_mean[mask].mean(axis=0)

        mu_hat = _mu_hat(Y_mean, omega)
        viols  = compute_violations(mu_hat, profile)
        dynamics.update(mu_hat, lambda_, profile)

        # ── Dual ascent ───────────────────────────────────────────────────────
        lambda_ = np.clip(lambda_ + alpha_lambda * viols, 0.0, lambda_max)

        # ── ELBO (approximate) ────────────────────────────────────────────────
        recon    = -sum(float(np.sum((Y_mean[i] - mu_role[omega[i]])**2))
                        for i in range(N)) / (N * sigma_sq)
        aug_elbo = recon - float(np.dot(lambda_, viols))
        elbo_trace.append(aug_elbo)

    mu_hat = _mu_hat(Y_mean, omega)
    viols  = compute_violations(mu_hat, profile)

    return CDMBDResult(
        omega=omega,
        rho_bb=dynamics.rho,
        lambda_=lambda_,
        violations=viols,
        mu_hat=mu_hat,
        elbo_trace=elbo_trace,
        n_B=int((omega == 1).sum()),
    )


# ─── Gap score ────────────────────────────────────────────────────────────────

def gap_score(Y: np.ndarray, profile: RequirementProfile, seed: int = 0) -> float:
    """
    Estimate data gap Δ(ω*) = score(ω*) − score(best_alt)
    using requirement-free inference (all lambda=0).
    """
    null = RequirementProfile([
        Requirement(r.channel, r.tau, 1e9, 0.0, r.name)
        for r in profile.reqs
    ])
    res    = run_cdmbd(Y, null, alpha_lambda=0.0, n_iter=30, seed=seed)
    Y_mean = Y.mean(axis=0)
    omega  = res.omega

    def _score(om: np.ndarray) -> float:
        mu_r = {r: Y_mean[om==r].mean(axis=0) if (om==r).sum()>0
                else np.zeros(Y.shape[2]) for r in range(3)}
        return -float(sum(np.sum((Y_mean[i] - mu_r[om[i]])**2)
                          for i in range(len(om))))

    base     = _score(omega)
    best_alt = -np.inf
    N = len(omega)
    for i in range(N):
        alt = omega.copy()
        if   omega[i] == 1: alt[i] = 2
        elif omega[i] == 2: alt[i] = 1
        else: continue
        if (alt==1).sum() < 1 or (alt==2).sum() < 1: continue
        s = _score(alt)
        if s > best_alt: best_alt = s

    return float(base - best_alt) if best_alt > -np.inf else 0.0


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def ensemble_run(Y: np.ndarray, profile: RequirementProfile,
                 n_runs: int = 6, **kwargs) -> List[CDMBDResult]:
    return [run_cdmbd(Y, profile, seed=i, **kwargs) for i in range(n_runs)]


def modal_result(results: List[CDMBDResult]) -> CDMBDResult:
    counts: Dict[int, int] = {}
    for r in results: counts[r.n_B] = counts.get(r.n_B, 0) + 1
    modal_nB = max(counts, key=counts.get)
    modal    = [r for r in results if r.n_B == modal_nB]
    return max(modal, key=lambda r: r.elbo_trace[-1] if r.elbo_trace else -np.inf)
