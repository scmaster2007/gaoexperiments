"""Code checks for core.py (CPU, small sizes)."""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

torch.set_default_dtype(torch.float32)
ok = True

xtr, ytr, xte, yte = core.load_cifar(os.path.expanduser(os.environ.get("CIFAR", "~/data")))
print("data", xtr.shape, xte.shape, "per-coord mean/var", xtr.mean().round(3), xtr.var().round(3))
X = torch.tensor(xtr[:40])

# 1. empirical NTK identity vs brute-force per-example gradients
for param in ("ntk", "standard"):
    m = core.MLP(3072, 64, 3, param=param, seed=1)
    K = core.empirical_ntk(m, X)
    params = dict(m.named_parameters())

    def f_single(p, x, k):
        return torch.func.functional_call(m, p, (x[None],))[0, k]

    Kb = torch.zeros(40, 40, dtype=torch.float64)
    for k in range(10):
        J = torch.func.vmap(torch.func.grad(lambda p, x: f_single(p, x, k)), in_dims=(None, 0))(params, X)
        G = torch.cat([J[n].reshape(40, -1) for n in params], 1).double()
        Kb += G @ G.T / 10
    err = ((K - Kb).norm() / Kb.norm()).item()
    print(f"[1] empirical NTK identity vs per-example gradients ({param}): rel err {err:.2e}")
    ok &= err < 1e-5

# 2. analytic kernel recursion vs Monte Carlo of the infinite-width expectations (depth 1 NNGP closed form check)
a = torch.randn(2, 50, dtype=torch.float64)
S, T = core.relu_kernels(a, a, depth=1, sw2=2.0, so2=1.0)
# depth 1: f = s_out * Wout relu(z1), z1 ~ N(0, Sigma1). NNGP = so2 * E[relu(u) relu(v)]
Sig1 = 2.0 * a @ a.T / 50
Lc = torch.linalg.cholesky(Sig1 + 1e-12 * torch.eye(2))
u = torch.randn(2_000_000, 2, dtype=torch.float64) @ Lc.T
mc = (torch.relu(u[:, :, None]) * torch.relu(u[:, None, :])).mean(0)
err = ((S - mc).abs().max() / S.abs().max()).item()
print(f"[2] ReLU NNGP closed form vs Monte Carlo: max rel err {err:.2e}")
ok &= err < 5e-3

# 3. save_depths consistency
Xs = X[:20].double()
_, T4 = core.relu_kernels(Xs, Xs, 4)
outs = core.relu_kernels(Xs, Xs, 6, save_depths=[2, 4, 6])
err = ((outs[4][1] - T4).norm() / T4.norm()).item()
print(f"[3] save_depths vs direct depth 4: rel err {err:.2e}")
ok &= err < 1e-12

# 4. empirical NTK -> analytic NTK as width grows (ntk parametrization), depth 3
for depth in (1, 3):
    _, Tinf = core.relu_kernels(X.double(), X.double(), depth)
    for w in (128, 512, 2048, 8192):
        errs = []
        for s in range(3):
            m = core.MLP(3072, w, depth, param="ntk", seed=s)
            errs.append(((core.empirical_ntk(m, X) - Tinf).norm() / Tinf.norm()).item())
        print(f"[4] depth {depth} width {w}: ||K_emp - K_inf|| / ||K_inf|| = {np.mean(errs):.4f}")

# 5. linear_gd_predict matches explicit GD on a linear model with the same kernel
n = 30
Ktt = core.relu_kernels(X[:n].double(), X[:n].double(), 2)[1]
Kst = core.relu_kernels(X[n:40].double(), X[:n].double(), 2)[1]
Y = torch.tensor(core.onehot(ytr[:n]), dtype=torch.float64)
f0t = torch.randn(n, 10, dtype=torch.float64) * 0.1
f0s = torch.randn(10, 10, dtype=torch.float64) * 0.1
eta = n / torch.linalg.eigvalsh(Ktt).max().item()
ft, fs = f0t.clone(), f0s.clone()
for _ in range(500):
    r = ft - Y
    ft = ft - eta / n * Ktt @ r
    fs = fs - eta / n * Kst @ r
pt, ps = core.linear_gd_predict(Ktt, Kst, f0t, f0s, Y, eta, 500, n)
err = max((pt - ft).abs().max().item(), (ps - fs).abs().max().item())
print(f"[5] closed-form linear GD vs explicit iteration: max abs err {err:.2e}")
ok &= err < 1e-8

# 6. determinism of MLP init
m1 = core.MLP(3072, 256, 4, seed=7)
m2 = core.MLP(3072, 256, 4, seed=7)
same = all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters()))
print(f"[6] same seed -> identical weights: {same}")
ok &= same
print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
