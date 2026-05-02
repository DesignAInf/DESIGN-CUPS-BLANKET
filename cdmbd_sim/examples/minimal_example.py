"""
examples/minimal_example.py
----------------------------
Minimal working example: run C-DMBD on a single cup and inspect the result.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cdmbd import generate, run_cdmbd, make_profile, compute_violations

# 1. Generate synthetic cup data (T=50 timesteps, N=30 nodes, D=7 channels)
Y, partition_true = generate('tea', N=30, T=50, seed=42)
print(f"Data shape: {Y.shape}   (T, N, D)")
print(f"True partition:  S={sum(partition_true==0)}, "
      f"B={sum(partition_true==1)}, Z={sum(partition_true==2)} nodes")

# 2. Define requirements (or load a preset)
profile = make_profile('tea')
print(f"\nRequirements for 'tea':")
for req in profile:
    print(f"  {req.name}: channel={req.channel}, tau={req.tau}, "
          f"eps={req.epsilon}, w={req.weight}")

# 3. Run one C-DMBD chain
result = run_cdmbd(Y, profile, n_iter=50, seed=0)
print(f"\nC-DMBD result:")
print(f"  |B| = {result.n_B} blanket nodes")
print(f"  ρ(A_bb) = {result.rho_bb:.3f}")
print(f"  ‖λ*‖₁  = {result.lambda_norm:.2f}")

# 4. Inspect binding constraints
print(f"\nBinding requirements (λ* > 0.1):")
for k, (req, lam, viol) in enumerate(
        zip(profile, result.lambda_, result.violations)):
    if lam > 0.1:
        print(f"  {req.name}: λ*={lam:.2f}, violation={viol:.3f}")

# 5. Inspect inferred topology
S_nodes = (result.omega == 0).sum()
B_nodes = (result.omega == 1).sum()
Z_nodes = (result.omega == 2).sum()
print(f"\nInferred topology: S={S_nodes}, B={B_nodes}, Z={Z_nodes}")
print(f"True topology:     S={sum(partition_true==0)}, "
      f"B={sum(partition_true==1)}, Z={sum(partition_true==2)}")
