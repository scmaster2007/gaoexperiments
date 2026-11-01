"""Adam in the NTK parametrization across width: full-batch Adam on a fixed CIFAR-10 subset, compared with
kernel predictions.

For each run (depth, width, seed, lr scaling p): bias-free ReLU MLP, NTK parametrization (core.MLP), full-batch
Adam (beta1 0.9, beta2 0.999, eps 1e-8), squared loss, lr = lr0 * (1024 / width)^p, `steps` steps.
Saved: train/test outputs at log-spaced steps; at init and at the end, on all n_train + n_test points:
  ntk      <grad f(x), grad f(x')>                         (output-averaged)
  asign    <grad f(x), sign grad f(x')>                    (A-SignGD kernel, Malladi et al. 2023)
  sign     <sign grad f(x), sign grad f(x')>               (SignGD kernel)
all exact, via the per-layer identity grad_W f = s * delta h^T, so sign(grad_W f) = sign(delta) sign(h)^T.
Also: infinite-width NTK; weight and kernel change; predictions of kernel regression (t -> infinity, network's own
f0 as offset) with each kernel at init, and with the infinite-width NTK.
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
p.add_argument("--data", default=os.path.expanduser("~/data"))
p.add_argument("--out", default=os.path.expanduser("~/adamwide"))
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--depth", type=int, default=4)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--p", type=float, default=0.0, help="lr = lr0 * (1024/width)^p")
p.add_argument("--lr0", type=float, default=1e-3)
p.add_argument("--n_train", type=int, default=2000)
p.add_argument("--n_test", type=int, default=1000)
p.add_argument("--steps", type=int, default=2000)
args = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)
L, w, s = args.depth, args.width, args.seed
lr = args.lr0 * (1024 / w) ** args.p
name = f"aw_L{L}_w{w}_p{args.p:g}_s{s}"
t0 = time.time()

xtr, ytr, xte, yte = core.load_cifar(args.data)
itr = core.balanced_subset(ytr, args.n_train, seed=0)
ite = core.balanced_subset(yte, args.n_test, seed=1)
Xall = torch.tensor(np.concatenate([xtr[itr], xte[ite]]), device=dev)
n = len(itr)
Y = torch.tensor(core.onehot(ytr[itr]), device=dev, dtype=torch.float64)


def kernels(model, X):
    """ntk, asign, sign kernels (output-averaged), exact for the bias-free MLP."""
    f, zs, hs = model(X.detach(), keep=True)
    N = X.shape[0]
    K = {k: torch.zeros(N, N, dtype=torch.float64, device=dev) for k in ("ntk", "asign", "sign")}
    hL = hs[-1].detach().double(); sL = torch.sign(hL)
    so = model.s_out
    # readout: grad wrt Wout[k] = s_out * h_L; identical for every output k
    K["ntk"] += so ** 2 * hL @ hL.T
    K["asign"] += so * hL @ sL.T
    K["sign"] += sL @ sL.T
    H = [h.detach().double() for h in hs[:-1]]
    SH = [torch.sign(h) for h in H]
    G = [(h @ h.T) for h in H]; GA = [(h @ sh.T) for h, sh in zip(H, SH)]; GS = [(sh @ sh.T) for sh in SH]
    for k in range(f.shape[1]):
        grads = torch.autograd.grad(f[:, k].sum(), zs, retain_graph=(k < f.shape[1] - 1))
        for l, d in enumerate(grads):
            d = d.detach().double(); sd = torch.sign(d); sc = model.scales[l]
            K["ntk"] += sc ** 2 * (d @ d.T) * G[l] / f.shape[1]
            K["asign"] += sc * (d @ sd.T) * GA[l] / f.shape[1]
            K["sign"] += (sd @ sd.T) * GS[l] / f.shape[1]
    return K


def krr(K, f0, ridge=1e-8):
    """t -> infinity kernel regression with offset f0: f0(test) + K_st K_tt^-1 (Y - f0(train)); a general
    (nonsymmetric) solve, so it also applies to the asymmetric kernel."""
    Ktt, Kst = K[:n, :n], K[n:, :n]
    A = Ktt + ridge * torch.diagonal(Ktt).abs().mean() * torch.eye(n, dtype=K.dtype, device=dev)
    return (f0[n:] + Kst @ torch.linalg.solve(A, Y - f0[:n])).cpu().numpy()


_, Kinf = core.relu_kernels(Xall, Xall, L)
model = core.MLP(3072, w, L, param="ntk", seed=s, device=dev)
with torch.no_grad():
    f0 = model(Xall).double()
K0 = kernels(model, Xall)
init_err = float((K0["ntk"] - Kinf).norm() / Kinf.norm())
preds = {f"krr_{k}_init": krr(K0[k], f0) for k in K0}
preds["krr_ntk_inf"] = krr(Kinf, f0)
preds["krr_ntk_inf_mean"] = krr(Kinf, torch.zeros_like(f0))
opt = torch.optim.Adam(model.parameters(), lr=lr)
ck = sorted(set([0] + [int(round(x)) for x in np.geomspace(1, args.steps, 15)]))
traj, losses = [], []
for step in range(args.steps + 1):
    if step in ck:
        with torch.no_grad():
            fa = model(Xall).double()
        traj.append(fa.cpu().numpy().astype(np.float32)); losses.append(float(0.5 * ((fa[:n] - Y) ** 2).sum(1).mean()))
    if step == args.steps:
        break
    opt.zero_grad(set_to_none=True)
    loss = 0.5 * ((model(Xall[:n]) - Y.float()) ** 2).sum(1).mean()
    loss.backward()
    opt.step()
    if not math.isfinite(loss.item()):
        break
KT = kernels(model, Xall)
g = torch.Generator(device="cpu").manual_seed(s)
dims = [3072] + [w] * L
num, den = 0.0, 0.0
for l in range(L):
    W0 = torch.randn(dims[l + 1], dims[l], generator=g).to(dev)
    num += float(((model.Ws[l].detach() - W0) ** 2).sum()); den += float((W0 ** 2).sum())
W0 = torch.randn(10, w, generator=g).to(dev)
num += float(((model.Wout.detach() - W0) ** 2).sum()); den += float((W0 ** 2).sum())
net = traj[-1][n:]
yt = yte[ite]


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


meta = {"depth": L, "width": w, "seed": s, "p": args.p, "lr": lr, "steps": args.steps, "n_train": n,
        "init_ntk_rel_err_vs_inf": init_err, "weight_rel_change": math.sqrt(num / den),
        "kernel_rel_change": {k: float((KT[k] - K0[k]).norm() / K0[k].norm()) for k in K0},
        "train_loss": dict(zip(map(str, ck), losses)), "test_acc_net": float((net.argmax(1) == yt).mean()),
        "test_rel_dist_net_vs": {k: rel(net, v) for k, v in preds.items()},
        "test_agree_net_vs": {k: float((net.argmax(1) == v.argmax(1)).mean()) for k, v in preds.items()},
        "test_acc_pred": {k: float((v.argmax(1) == yt).mean()) for k, v in preds.items()},
        "seconds": time.time() - t0}
arr = {"ckpts": np.array(ck), "traj": np.stack(traj), "y_test": yt, "y_train": ytr[itr]}
arr.update({k: v.astype(np.float32) for k, v in preds.items()})
for k in K0:
    arr[f"K0_{k}"] = K0[k].float().cpu().numpy(); arr[f"KT_{k}"] = KT[k].float().cpu().numpy()
arr["Kinf_ntk"] = Kinf.float().cpu().numpy()
np.savez_compressed(os.path.join(args.out, name + ".npz"), **arr)
json.dump(meta, open(os.path.join(args.out, name + ".json"), "w"))
print(time.strftime("%H:%M:%S"), name, json.dumps({k: v for k, v in meta.items() if not isinstance(v, dict)}),
      json.dumps(meta["test_rel_dist_net_vs"]), flush=True)
