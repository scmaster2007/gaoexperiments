"""H2 (from "On the implicit bias in deep-learning algorithms"): in overparametrized linear regression, GD
converges to the interpolant closest to its initialization w0 in L2. Does Adam? And is Adam's choice set by
the loss landscape, or by the optimizer and its coordinate system?

Setup: n=30 points, d=200 dimensions, a sparse or dense linear teacher, 5 seeds. Every optimizer is run to
an exact interpolant (cosine-decayed learning rate). For each solution w we report:
    dist_to_gd     ||w - w_gd|| / ||w_gd - w0||, where w_gd = w0 + pinv(X)(y - X w0) is GD's theoretical limit
    dist_to_signgd same, relative to the sign-GD solution (do the adaptive methods share one bias?)
    l2/l1/linf     its norm divided by the smallest such norm of any interpolant
    rot_disagree   rotation test: train on X Q (Q random orthogonal, init Q^T w0). The landscape is the same up
                   to rotation, so a landscape-determined bias gives the same test predictions (0 here).
    test_mse       error against the teacher on fresh points

    python3 experiments/h2_adaptive_implicit_bias/linreg_bias.py
"""
import os
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
N, D, N_TEST, STEPS = 30, 200, 2000, 20000
OPTIMIZERS = [("gd", 0.05), ("sgd", 0.02), ("signgd", 1e-3), ("adagrad", 0.05), ("rmsprop", 1e-3),
              ("adam", 1e-3), ("adam", 1e-2), ("adam", 1e-1), ("adam_beta2=0.9", 1e-2)]


def min_l1_interpolant(X, y):
    """min ||w||_1 s.t. Xw = y, written as w = u - v with u, v >= 0."""
    r = linprog(np.ones(2 * D), A_eq=np.hstack([X, -X]), b_eq=y, bounds=(0, None), method="highs")
    return r.x[:D] - r.x[D:]


def min_linf_interpolant(X, y):
    """min t s.t. Xw = y and -t <= w_i <= t."""
    I, ones = np.eye(D), np.ones((D, 1))
    r = linprog(np.r_[np.zeros(D), 1.0],
                A_ub=np.block([[I, -ones], [-I, -ones]]), b_ub=np.zeros(2 * D),
                A_eq=np.hstack([X, np.zeros((N, 1))]), b_eq=y,
                bounds=[(None, None)] * D + [(0, None)], method="highs")
    return r.x[:D]


def make_optimizer(name, w, lr):
    if name in ("gd", "sgd"):
        return torch.optim.SGD([w], lr=lr)
    if name == "adagrad":
        return torch.optim.Adagrad([w], lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop([w], lr=lr)
    if name == "adam":
        return torch.optim.Adam([w], lr=lr)
    if name == "adam_beta2=0.9":
        return torch.optim.Adam([w], lr=lr, betas=(0.9, 0.9))
    return None                                                        # signgd is applied by hand


def train(X, y, w0, name, lr, seed):
    """Run one optimizer to an interpolant of Xw = y, starting at w0."""
    X, y = torch.tensor(X), torch.tensor(y)
    w = torch.tensor(w0).clone().requires_grad_(True)
    opt = make_optimizer(name, w, lr)
    gen = torch.Generator().manual_seed(seed)
    for t in range(STEPS):
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / STEPS))
        rows = torch.randperm(N, generator=gen)[:5] if name == "sgd" else slice(None)
        loss = 0.5 * ((X[rows] @ w - y[rows]) ** 2).mean()
        if name == "signgd":
            grad, = torch.autograd.grad(loss, w)
            with torch.no_grad():
                w -= lr_t * grad.sign()
        else:
            for group in opt.param_groups:
                group["lr"] = lr_t
            opt.zero_grad()
            loss.backward()
            opt.step()
    return w.detach().numpy()


def make_problem(seed, teacher):
    rng = np.random.default_rng(seed)
    X, X_test = rng.standard_normal((N, D)), rng.standard_normal((N_TEST, D))
    w_star = np.zeros(D)
    if teacher == "sparse":
        w_star[rng.choice(D, 5, replace=False)] = 3 * rng.standard_normal(5)
    else:
        w_star = np.sqrt(45 / D) * rng.standard_normal(D)               # same expected ||w*||^2 as sparse
    w0 = 0.1 * rng.standard_normal(D)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    return X, X @ w_star, X_test, X_test @ w_star, w0, Q


rows = []
for seed in range(5):
    for teacher in ("sparse", "dense"):
        X, y, X_test, y_test, w0, Q = make_problem(seed, teacher)
        w_gd = w0 + np.linalg.pinv(X) @ (y - X @ w0)
        min_l1 = np.abs(min_l1_interpolant(X, y)).sum()
        min_linf = np.abs(min_linf_interpolant(X, y)).max()
        scale = np.linalg.norm(w_gd - w0)
        w_sign = train(X, y, w0, "signgd", 1e-3, seed)
        for name, lr in OPTIMIZERS:
            w = train(X, y, w0, name, lr, seed)
            w_rot = train(X @ Q, y, Q.T @ w0, name, lr, seed)
            pred, pred_rot = X_test @ w, X_test @ Q @ w_rot
            rows.append(dict(
                seed=seed, teacher=teacher, opt=name, lr=lr,
                train_rmse=np.sqrt(np.mean((X @ w - y) ** 2)),
                dist_to_gd=np.linalg.norm(w - w_gd) / scale,
                dist_to_signgd=np.linalg.norm(w - w_sign) / scale,
                l2_ratio=np.linalg.norm(w - w0) / scale,
                l1_ratio=np.abs(w).sum() / min_l1,
                linf_ratio=np.abs(w).max() / min_linf,
                rot_disagree=np.sqrt(np.mean((pred - pred_rot) ** 2) / np.mean(pred ** 2)),
                test_mse=np.mean((pred - y_test) ** 2)))
            print(f"seed {seed} {teacher:6s} {name:15s} lr {lr:<6g} dist_to_gd {rows[-1]['dist_to_gd']:.3f}", flush=True)

results = pd.DataFrame(rows)
results.to_csv(os.path.join(HERE, "results.csv"), index=False)
summary = results.drop(columns="seed").groupby(["teacher", "opt", "lr"], sort=False).mean().round(3)
summary.to_csv(os.path.join(HERE, "summary.csv"))
print(summary.to_string())
