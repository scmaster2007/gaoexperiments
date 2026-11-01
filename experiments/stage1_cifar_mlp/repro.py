"""Reproductions of known wide/deep-network results, plus the Adam epoch pilot.

Subcommands
  r12     R1 + R2 for one (depth, width, seed): empirical NTK at init vs infinite-width NTK (Jacot et al. 2018);
          full-batch GD in NTK parametrization vs the linearized / infinite-width predictions, and how much the
          weights and the kernel move (Lee et al. 2019).
  r3sig   R3a: forward and backward signal propagation at initialization across depth (He et al. 2015;
          Poole et al. 2016), standard parametrization, several weight variances.
  r3train R3b: trainability of deep ReLU MLPs vs weight variance and depth with SGD (Schoenholz et al. 2017 style).
  r4      R4: infinite-width NTK/NNGP of deep ReLU MLPs vs depth on CIFAR-10 (Xiao, Pennington, Schoenholz 2020).
  r4full  Infinite-width NTK/NNGP regression on the full 50k CIFAR-10 training set, a few depths.
  pilot   Adam (PyTorch defaults), realistic MLP, batch 128: training-loss curves to set the epoch count.
          Saves loss/accuracy curves only.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

p = argparse.ArgumentParser()
p.add_argument("cmd")
p.add_argument("--data", default=os.path.expanduser("~/data"))
p.add_argument("--out", default=os.path.expanduser("~/results"))
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--depth", type=int, default=4)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_train", type=int, default=2000)
p.add_argument("--n_test", type=int, default=1000)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--lr_frac", type=float, default=0.5, help="GD step as a fraction of 2 / lambda_max(K_inf / n)")
p.add_argument("--sw2", type=float, default=2.0)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--depths", default="")
p.add_argument("--tag", default="")
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)


def save(name, meta, arrays=None):
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    if arrays:
        np.savez_compressed(os.path.join(args.out, name + ".npz"), **arrays)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


xtr, ytr, xte, yte = core.load_cifar(args.data)


# ============================================================================= R1 + R2

def cmd_r12():
    L, w, s = args.depth, args.width, args.seed
    name = f"r12_L{L}_w{w}_s{s}"
    itr = core.balanced_subset(ytr, args.n_train, seed=0)
    ite = core.balanced_subset(yte, args.n_test, seed=1)
    Xall = torch.tensor(np.concatenate([xtr[itr], xte[ite]]), device=dev)
    n, m = len(itr), len(ite)
    Y = torch.tensor(core.onehot(ytr[itr]), device=dev, dtype=torch.float64)
    t0 = time.time()

    # infinite-width kernels (float64), shared by all widths/seeds at this depth
    Kinf_nngp, Kinf = core.relu_kernels(Xall, Xall, L)
    Ktt, Kst = Kinf[:n, :n], Kinf[n:, :n]
    lam_max = torch.linalg.eigvalsh(Ktt)[-1].item()
    eta = args.lr_frac * 2.0 * n / lam_max
    log(name, f"eta={eta:.4f} lam_max/n={lam_max / n:.3f}")

    model = core.MLP(3072, w, L, param="ntk", seed=s, device=dev)
    P = sum(q.numel() for q in model.parameters())
    with torch.no_grad():
        f0 = model(Xall).double()
    K0 = core.empirical_ntk(model, Xall)
    init_err = ((K0 - Kinf).norm() / Kinf.norm()).item()
    init_err_tt = ((K0[:n, :n] - Ktt).norm() / Ktt.norm()).item()
    log(name, f"P={P / 1e6:.1f}M init kernel err {init_err:.4f} ({time.time() - t0:.0f}s)")

    # full-batch GD on the training subset
    Xtr = Xall[:n]
    Ytr = Y.float()
    ckpts = sorted(set([0] + [int(round(x)) for x in np.geomspace(1, args.steps, 12)]))
    traj = {}
    losses = {}
    for step in range(args.steps + 1):
        if step in ckpts:
            with torch.no_grad():
                fa = model(Xall).double()
            traj[step] = fa.cpu().numpy().astype(np.float32)
            losses[step] = (0.5 * ((fa[:n] - Y) ** 2).sum(1).mean()).item()
        if step == args.steps:
            break
        model.zero_grad(set_to_none=True)
        f = model(Xtr)
        loss = 0.5 * ((f - Ytr) ** 2).sum(1).mean()
        loss.backward()
        with torch.no_grad():
            for q in model.parameters():
                q -= eta * q.grad
        if not math.isfinite(loss.item()):
            log(name, "non-finite loss at step", step)
            break
    log(name, f"trained {args.steps} steps, loss {losses[max(losses)]:.5f} ({time.time() - t0:.0f}s)")
    fT = torch.tensor(traj[max(traj)], device=dev, dtype=torch.float64)
    KT = core.empirical_ntk(model, Xall)
    k_change = ((KT - K0).norm() / K0.norm()).item()

    # weight change, regenerating the initial weights from the seed (avoids storing a copy)
    g = torch.Generator(device="cpu").manual_seed(s)
    dims = [3072] + [w] * L
    num2, den2, per_layer = 0.0, 0.0, []
    for l in range(L):
        W0 = torch.randn(dims[l + 1], dims[l], generator=g).to(dev)
        d2 = ((model.Ws[l].detach() - W0) ** 2).sum().item()
        n2 = (W0 ** 2).sum().item()
        per_layer.append(math.sqrt(d2 / n2))
        num2 += d2; den2 += n2
        del W0
    W0 = torch.randn(10, w, generator=g).to(dev)
    d2 = ((model.Wout.detach() - W0) ** 2).sum().item(); n2 = (W0 ** 2).sum().item()
    per_layer.append(math.sqrt(d2 / n2)); num2 += d2; den2 += n2
    w_change = math.sqrt(num2 / den2)

    # linearized predictions at every checkpoint: own kernel at init (K0), and the infinite-width kernel
    preds = {}
    for kname, K in (("lin0", K0), ("lininf", Kinf)):
        for step in ckpts:
            ft, fs = core.linear_gd_predict(K[:n, :n], K[n:, :n], f0[:n], f0[n:], Y, eta, step, n)
            preds[(kname, step)] = torch.cat([ft, fs]).cpu().numpy().astype(np.float32)
    # infinite-width mean predictor (f0 = 0), which the seed average should approach
    zero = torch.zeros_like(f0)
    mean_inf = {}
    for step in ckpts:
        ft, fs = core.linear_gd_predict(Ktt, Kst, zero[:n], zero[n:], Y, eta, step, n)
        mean_inf[step] = torch.cat([ft, fs]).cpu().numpy().astype(np.float32)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    yt = yte[ite]
    T = max(traj)
    net = traj[T]
    out = {
        "depth": L, "width": w, "seed": s, "params": P, "n_train": n, "n_test": m, "steps": args.steps,
        "eta": eta, "lam_max_over_n": lam_max / n,
        "init_kernel_rel_err": init_err, "init_kernel_rel_err_train": init_err_tt,
        "kernel_rel_change": k_change, "weight_rel_change": w_change, "weight_rel_change_per_layer": per_layer,
        "train_loss": {str(k): v for k, v in losses.items()},
        "ckpts": ckpts,
        "test_rel_dist_net_vs_lin0": rel(net[n:], preds[("lin0", T)][n:]),
        "test_rel_dist_net_vs_lininf": rel(net[n:], preds[("lininf", T)][n:]),
        "test_agree_net_vs_lin0": float((net[n:].argmax(1) == preds[("lin0", T)][n:].argmax(1)).mean()),
        "test_agree_net_vs_lininf": float((net[n:].argmax(1) == preds[("lininf", T)][n:].argmax(1)).mean()),
        "test_acc_net": float((net[n:].argmax(1) == yt).mean()),
        "test_acc_lininf": float((preds[("lininf", T)][n:].argmax(1) == yt).mean()),
        "test_acc_meaninf": float((mean_inf[T][n:].argmax(1) == yt).mean()),
        "traj_rel_dist_net_vs_lininf": {str(k): rel(traj[k][n:], preds[("lininf", k)][n:]) for k in ckpts if k > 0},
        "traj_rel_dist_net_vs_lin0": {str(k): rel(traj[k][n:], preds[("lin0", k)][n:]) for k in ckpts if k > 0},
        "seconds": time.time() - t0,
    }
    save(name, out, {
        "traj": np.stack([traj[k] for k in ckpts]),
        "lin0": np.stack([preds[("lin0", k)] for k in ckpts]),
        "lininf": np.stack([preds[("lininf", k)] for k in ckpts]),
        "meaninf": np.stack([mean_inf[k] for k in ckpts]),
        "ckpts": np.array(ckpts), "y_test": yt, "y_train": ytr[itr],
    })
    log(name, json.dumps({k: v for k, v in out.items() if not isinstance(v, (dict, list))}))


# ============================================================================= R3a: signal propagation

def cmd_r3sig():
    depth = args.depth
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xte), 256, replace=False)
    X = torch.tensor(xte[idx], device=dev)
    pairs = rng.choice(256, size=(200, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    res = []
    for sw2 in [1.0, 1.5, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]:
        for w in [1024, 4096]:
            for s in range(5):
                model = core.MLP(3072, w, depth, param="standard", sw2=sw2, so2=1.0, seed=s, device=dev)
                X.requires_grad_(False)
                f, zs, hs = model(X, keep=True)
                q_emp = [(z.double() ** 2).mean().item() for z in zs]
                c_emp = []
                for z in zs:
                    z = z.double()
                    a, b = z[pairs[:, 0]], z[pairs[:, 1]]
                    c_emp.append(((a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1))).mean().item())
                grads = torch.autograd.grad(f[:, 0].sum(), zs)
                g_emp = [(gz.double() ** 2).sum(1).mean().item() for gz in grads]
                res.append({"sw2": sw2, "width": w, "seed": s, "q": q_emp, "c": c_emp, "g": g_emp})
                del f, zs, hs, grads, model
        log("r3sig sw2", sw2)
    # theory
    Xd = X.double()
    q1 = (Xd ** 2).sum(1).mean().item() / 3072  # mean ||x||^2 / d
    a, b = Xd[pairs[:, 0]], Xd[pairs[:, 1]]
    c = ((a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1))).cpu().numpy()
    c_theory = []
    for l in range(depth):
        c_theory.append(float(c.mean()))
        th = np.arccos(np.clip(c, -1, 1))
        c = (np.sqrt(1 - c ** 2) + (np.pi - th) * c) / np.pi
    save(f"r3sig_L{depth}", {"results": res, "x_sq_over_d": q1, "c_theory": c_theory, "depth": depth,
                             "note": "q_theory(l) = sw2 * x_sq_over_d * (sw2/2)^(l-1), l = 1..depth; "
                                     "g_theory(l) = (so2/2) * (sw2/2)^(depth-l), so2 = 1"})


# ============================================================================= R3b: trainability

def cmd_r3train():
    L, sw2, lr = args.depth, args.sw2, args.lr
    name = f"r3train_L{L}_sw{sw2}_lr{lr}"
    X = torch.tensor(xtr, device=dev)
    Y = torch.tensor(core.onehot(ytr), device=dev)
    Xt = torch.tensor(xte, device=dev)
    model = core.MLP(3072, 512, L, param="standard", sw2=sw2, so2=1.0, seed=0, device=dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    g = torch.Generator(device=dev).manual_seed(0)
    hist = []
    finite = True
    for ep in range(args.epochs):
        perm = torch.randperm(len(X), generator=g, device=dev)
        tot = 0.0
        for i in range(0, len(X), args.batch):
            b = perm[i:i + args.batch]
            loss = 0.5 * ((model(X[b]) - Y[b]) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        tot /= len(X)
        finite = math.isfinite(tot)
        with torch.no_grad():
            acc = float(torch.cat([model(X[i:i + 5000]).argmax(1) for i in range(0, len(X), 5000)])
                        .eq(torch.tensor(ytr, device=dev)).float().mean())
        hist.append({"epoch": ep + 1, "train_loss": tot, "train_acc": acc})
        if not finite:
            break
    with torch.no_grad():
        tacc = float(model(Xt).argmax(1).eq(torch.tensor(yte, device=dev)).float().mean()) if finite else float("nan")
    save(name, {"depth": L, "sw2": sw2, "lr": lr, "width": 512, "epochs": args.epochs, "batch": args.batch,
                "optimizer": "SGD momentum 0.9", "history": hist, "test_acc": tacc, "finite": finite})
    log(name, hist[-1], tacc)


# ============================================================================= R4: kernels vs depth

def relu_step(S, T, q1, q2, scale):
    norm = torch.sqrt(q1[:, None] * q2[None, :])
    c = torch.clamp(S / norm, -1.0, 1.0)
    th = torch.arccos(c)
    Ephi = norm * (torch.sin(th) + (math.pi - th) * c) / (2 * math.pi)
    Edphi = (math.pi - th) / (2 * math.pi)
    return scale * Ephi, scale * Ephi + scale * Edphi * T


def cmd_r4():
    depths = [int(x) for x in args.depths.split(",")]
    itr = core.balanced_subset(ytr, args.n_train, seed=0)
    Xtr = torch.tensor(xtr[itr], device=dev, dtype=torch.float64)
    Xte = torch.tensor(xte, device=dev, dtype=torch.float64) if args.n_test >= len(xte) else \
        torch.tensor(xte[core.balanced_subset(yte, args.n_test, seed=1)], device=dev, dtype=torch.float64)
    yt = yte if args.n_test >= len(xte) else yte[core.balanced_subset(yte, args.n_test, seed=1)]
    Y = torch.tensor(core.onehot(ytr[itr]), device=dev, dtype=torch.float64)
    d, n = 3072, len(itr)
    sw2, so2 = 2.0, 1.0
    Stt = sw2 * Xtr @ Xtr.T / d
    Sst = sw2 * Xte @ Xtr.T / d
    qtr = sw2 * (Xtr ** 2).sum(1) / d
    qte = sw2 * (Xte ** 2).sum(1) / d
    Ttt, Tst = Stt.clone(), Sst.clone()
    rng = np.random.default_rng(0)
    subsets = {m: [rng.choice(n, m, replace=False) for _ in range(100)] for m in (12, 20)}
    # Gaussian inputs for the condition-number check (unit-norm rows)
    G = torch.randn(20, d, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).to(dev)
    G = G / G.norm(dim=1, keepdim=True) * math.sqrt(d)
    Sg = sw2 * G @ G.T / d; qg = sw2 * (G ** 2).sum(1) / d; Tg = Sg.clone()
    res = []
    for l in range(1, max(depths) + 1):
        if l in depths:
            row = {"depth": l}
            nn_tt, nt_tt = relu_step(Stt, Ttt, qtr, qtr, so2)
            nn_st, nt_st = relu_step(Sst, Tst, qte, qtr, so2)
            for kname, Kt, Ks in (("nngp", nn_tt, nn_st), ("ntk", nt_tt, nt_st)):
                diag = torch.diagonal(Kt)
                off = (Kt.sum() - diag.sum()) / (n * n - n)
                row[f"{kname}_diag_mean"] = diag.mean().item()
                row[f"{kname}_offdiag_mean"] = off.item()
                kap = {}
                for m, subs in subsets.items():
                    ks = []
                    for sub in subs:
                        ev = torch.linalg.eigvalsh(Kt[sub][:, sub])
                        ks.append((ev[-1] / ev[0]).item())
                    kap[m] = float(np.median(ks))
                row[f"{kname}_kappa_cifar_m12"] = kap[12]
                row[f"{kname}_kappa_cifar_m20"] = kap[20]
                for rr in (1e-8, 1e-6, 1e-4, 1e-2):
                    try:
                        pred = core.kernel_regression(Kt, Ks, Y, ridge_rel=rr)
                    except Exception as e:  # Cholesky failure
                        row[f"{kname}_acc_ridge{rr:g}"] = None
                        continue
                    row[f"{kname}_acc_ridge{rr:g}"] = float((pred.argmax(1).cpu().numpy() == yt).mean())
                    row[f"{kname}_mse_ridge{rr:g}"] = float(((pred - torch.tensor(core.onehot(yt), device=dev,
                                                           dtype=torch.float64)) ** 2).sum(1).mean())
                    row[f"{kname}_pred_input_std_ridge{rr:g}"] = float(pred.std(0).mean())
            gnn, gnt = relu_step(Sg, Tg, qg, qg, so2)
            for kname, Kg in (("nngp", gnn), ("ntk", gnt)):
                ev = torch.linalg.eigvalsh(Kg[:12, :12]); row[f"{kname}_kappa_gauss_m12"] = (ev[-1] / ev[0]).item()
                ev = torch.linalg.eigvalsh(Kg); row[f"{kname}_kappa_gauss_m20"] = (ev[-1] / ev[0]).item()
            res.append(row)
            log("r4", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                       if "acc" in k or k == "depth" or "kappa_cifar_m12" in k})
            save(f"r4_n{n}{args.tag}", {"n_train": n, "n_test": len(yt), "rows": res})
        Stt, Ttt = relu_step(Stt, Ttt, qtr, qtr, sw2)
        Sst, Tst = relu_step(Sst, Tst, qte, qtr, sw2)
        Sg, Tg = relu_step(Sg, Tg, qg, qg, sw2)
        qtr, qte, qg = sw2 * qtr / 2, sw2 * qte / 2, sw2 * qg / 2


def cmd_r4full():
    """Full 50k training set, one depth at a time, kernels built in row blocks (float64)."""
    depths = [int(x) for x in args.depths.split(",")]
    Xtr = torch.tensor(xtr, device=dev, dtype=torch.float64)
    Xte = torch.tensor(xte, device=dev, dtype=torch.float64)
    Y = torch.tensor(core.onehot(ytr), device=dev, dtype=torch.float64)
    Yt = torch.tensor(core.onehot(yte), device=dev, dtype=torch.float64)
    n = len(Xtr)
    res = []
    for L in depths:
        for kidx, kname in ((1, "ntk"), (0, "nngp")):
            t0 = time.time()
            K = torch.empty(n, n, dtype=torch.float64, device=dev)
            for i in range(0, n, 5000):
                K[i:i + 5000] = core.relu_kernels(Xtr[i:i + 5000], Xtr, L)[kidx]
            Ks = torch.empty(len(Xte), n, dtype=torch.float64, device=dev)
            for i in range(0, len(Xte), 5000):
                Ks[i:i + 5000] = core.relu_kernels(Xte[i:i + 5000], Xtr, L)[kidx]
            row = {"depth": L, "kernel": kname}
            mean_diag = torch.diagonal(K).mean()
            added = 0.0
            for rr in (1e-6, 1e-4, 1e-2):
                K.diagonal().add_((rr - added) * mean_diag)  # cumulative ridge = rr * mean diagonal
                added = rr
                try:
                    Lc = torch.linalg.cholesky(K)
                    pred = Ks @ torch.cholesky_solve(Y, Lc)
                    del Lc
                    row[f"acc_ridge{rr:g}"] = float((pred.argmax(1).cpu().numpy() == yte).mean())
                    row[f"mse_ridge{rr:g}"] = float(((pred - Yt) ** 2).sum(1).mean())
                except Exception as e:
                    row[f"acc_ridge{rr:g}"] = None
                torch.cuda.empty_cache()
            row["seconds"] = time.time() - t0
            res.append(row)
            log("r4full", row)
            save("r4full", {"n_train": n, "n_test": len(Xte), "rows": res})
            del K, Ks
            torch.cuda.empty_cache()


# ============================================================================= Adam epoch pilot

def cmd_pilot():
    L, w = args.depth, args.width
    name = f"pilot_L{L}_w{w}_b{args.batch}_s{args.seed}"
    X = torch.tensor(xtr, device=dev)
    Y = torch.tensor(core.onehot(ytr), device=dev)
    Xt = torch.tensor(xte, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    yte_t = torch.tensor(yte, device=dev)
    model = core.realistic_mlp(3072, w, L, seed=args.seed, device=dev)
    opt = torch.optim.Adam(model.parameters())  # PyTorch defaults
    g = torch.Generator(device=dev).manual_seed(args.seed)
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(len(X), generator=g, device=dev)
        tot = torch.zeros((), device=dev)
        for i in range(0, len(X), args.batch):
            b = perm[i:i + args.batch]
            loss = 0.5 * ((model(X[b]) - Y[b]) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(b)
        row = {"epoch": ep, "running_train_loss": (tot / len(X)).item()}
        if ep % 5 == 0 or ep <= 5:
            with torch.no_grad():
                ftr = torch.cat([model(X[i:i + 10000]) for i in range(0, len(X), 10000)])
                row["train_loss"] = (0.5 * ((ftr - Y) ** 2).sum(1).mean()).item()
                row["train_acc"] = ftr.argmax(1).eq(ytr_t).float().mean().item()
                row["test_acc"] = model(Xt).argmax(1).eq(yte_t).float().mean().item()
        hist.append(row)
        if ep % 25 == 0:
            log(name, row, f"{time.time() - t0:.0f}s")
            save(name, {"depth": L, "width": w, "batch": args.batch, "optimizer": "Adam (PyTorch defaults)",
                        "history": hist})
    save(name, {"depth": L, "width": w, "batch": args.batch, "optimizer": "Adam (PyTorch defaults)",
                "history": hist, "seconds": time.time() - t0})


{"r12": cmd_r12, "r3sig": cmd_r3sig, "r3train": cmd_r3train, "r4": cmd_r4, "r4full": cmd_r4full,
 "pilot": cmd_pilot}[args.cmd]()
