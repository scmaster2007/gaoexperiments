"""Kernel trajectories for Adam vs SGD. Same model, data, initialization, data order, loss and optimizer
settings as rich.py; only the logging differs.

At every kernel checkpoint (epochs 0, 1, 2, 3, 5, 7, then every 10 epochs to the end), on the same 500 probe points
as rich.py (250 train + 250 test, fixed for every run), it saves:
  K_std   output-averaged empirical NTK, (1/10) sum_k J_k J_k^T, over all weights and biases
  K_adam  (Adam only) (1/10) sum_k J_k diag(q) J_k^T with q = 1 / (sqrt(vhat) + eps), Adam's denominator at that step
          (vhat = exp_avg_sq / (1 - beta2^t)); the kernel of one Adam step with the momentum replaced by the current
          gradient. Not defined at step 0 (no optimizer state yet).
  K_live  (Adam only) the same with q set to 0 on coordinates where sqrt(vhat) < eps
plus train/test loss and accuracy, dead-unit fraction per hidden layer on the kernel probe points, and (Adam) the
fraction of coordinates with sqrt(vhat) < eps. Top Hessian eigenvalue of the training loss on the same 2048 fixed
training examples as rich.py at epochs 0, 1, 3, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500.
Each checkpoint is written to its own file as soon as it is computed, so a stopped run keeps everything before it.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1_cifar_mlp"))  # core.py
import core

p = argparse.ArgumentParser()
p.add_argument("--data", default=os.path.expanduser("~/data"))
p.add_argument("--out", default=os.path.expanduser("~/kern"))
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--depth", type=int, default=4)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--opt", default="adam", choices=["adam", "sgd"])
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eps", type=float, default=1e-8)
p.add_argument("--beta1", type=float, default=0.9)
p.add_argument("--beta2", type=float, default=0.999)
p.add_argument("--momentum", type=float, default=0.9)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--epochs", type=int, default=500)
p.add_argument("--small", action="store_true", help="CPU test mode: tiny subsets")
p.add_argument("--max_epochs_run", type=int, default=-1, help="stop after this many epochs (timing tests)")
p.add_argument("--chunk_floats", type=float, default=1.2e8, help="max floats per per-example-gradient block")
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device(args.device)
name = f"kern_{args.opt}_L{args.depth}_w{args.width}_b{args.batch}_lr{args.lr:g}_eps{args.eps:g}_const{args.epochs}_s{args.seed}"
rdir = os.path.join(args.out, name)
os.makedirs(rdir, exist_ok=True)
t_start = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), name, *a, flush=True)


# ----------------------------------------------------------------------------- data (as rich.py)
xtr, ytr, xte, yte = core.load_cifar(args.data)
if args.small:
    xtr, ytr, xte, yte = xtr[:2000], ytr[:2000], xte[:500], yte[:500]
X = torch.tensor(xtr, device=dev)
Y = torch.tensor(core.onehot(ytr), device=dev)
Xt = torch.tensor(xte, device=dev)
Yt = torch.tensor(core.onehot(yte), device=dev)
ytr_t = torch.tensor(ytr, device=dev)
yte_t = torch.tensor(yte, device=dev)
N = len(X)
rng = np.random.default_rng(12345)  # same draws, in the same order, as rich.py
probe_train_out = np.sort(rng.choice(N, min(2000, N), replace=False))
probe_rep = np.sort(rng.choice(len(Xt), min(1000, len(Xt)), replace=False))
probe_ntk_tr = np.sort(rng.choice(N, 250, replace=False))
probe_ntk_te = np.sort(rng.choice(len(Xt), 250, replace=False))
probe_hess = np.sort(rng.choice(N, min(2048, N), replace=False))
X_ntk = torch.cat([X[probe_ntk_tr], Xt[probe_ntk_te]])
X_h, Y_h = X[probe_hess], Y[probe_hess]

# ----------------------------------------------------------------------------- model / optimizer (as rich.py)
model = core.realistic_mlp(3072, args.width, args.depth, seed=args.seed, device=dev)
linears = [m for m in model if isinstance(m, torch.nn.Linear)]
params = list(model.parameters())
if args.opt == "adam":
    opt = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
else:
    opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
spe = math.ceil(N / args.batch)
total_steps = spe * args.epochs


def forward_all(Xs, chunk=10000):
    with torch.no_grad():
        return torch.cat([model(Xs[i:i + chunk]) for i in range(0, len(Xs), chunk)])


def loss_fn(f, y):
    return 0.5 * ((f - y) ** 2).sum(1).mean()


# ----------------------------------------------------------------------------- kernels
def adam_q():
    """Per-parameter q = 1/(sqrt(vhat)+eps) and live mask, in the order of `params`; None before the first step."""
    st = [opt.state[q] for q in params]
    if args.opt != "adam" or len(st[0]) == 0:
        return None, None, float("nan")
    t = int(st[0]["step"])
    b2c = 1 - args.beta2 ** t
    sv = [(s["exp_avg_sq"] / b2c).sqrt() for s in st]
    q = [1.0 / (x + args.eps) for x in sv]
    live = [(x >= args.eps).to(x.dtype) for x in sv]
    n_tot = sum(x.numel() for x in sv)
    frac_below = sum(float((x < args.eps).sum()) for x in sv) / n_tot
    return q, live, frac_below


def kernels(Xs):
    """Returns K_std and, for Adam with optimizer state, K_adam and K_live (float64 numpy, n x n)."""
    n = Xs.shape[0]
    qs, lives, frac_below = adam_q()
    h = Xs
    zs, hs = [], []
    for l, lin in enumerate(linears):
        hs.append(h.detach())
        z = lin(h)
        zs.append(z)
        h = torch.relu(z) if l < len(linears) - 1 else z
    f = zs[-1]
    nout = f.shape[1]
    K = torch.zeros(n, n, dtype=torch.float64, device=dev)
    KA = torch.zeros_like(K) if qs is not None else None
    KL = torch.zeros_like(K) if qs is not None else None
    G = [(hh.double() @ hh.double().T + 1.0) for hh in hs]  # <h, h'> + 1 (bias) per layer
    for k in range(nout):
        D = torch.autograd.grad(f[:, k].sum(), zs, retain_graph=(k < nout - 1))
        for l, d in enumerate(D):
            d = d.detach()
            K += (d.double() @ d.double().T) * G[l]
            if qs is None:
                continue
            qW, qb = qs[2 * l], qs[2 * l + 1]          # params order: W0, b0, W1, b1, ...
            lW, lb = lives[2 * l], lives[2 * l + 1]
            Hl = hs[l]
            out_l, in_l = qW.shape
            # bias part: sum_a d_ia d_ja q_a
            KA += ((d * qb) @ d.T).double()
            KL += ((d * (qb * lb)) @ d.T).double()
            # weight part: sum_{a,b} d_ia h_ib d_ja h_jb q_ab, in blocks of rows a
            c = max(1, int(args.chunk_floats // (n * in_l)))
            for a0 in range(0, out_l, c):
                a1 = min(out_l, a0 + c)
                A = (d[:, a0:a1, None] * Hl[:, None, :]).reshape(n, -1)
                qa = qW[a0:a1].reshape(-1)
                KA += ((A * qa) @ A.T).double()
                KL += ((A * (qa * lW[a0:a1].reshape(-1))) @ A.T).double()
                del A
    out = {"K_std": (K / nout).cpu().numpy()}
    if qs is not None:
        out["K_adam"] = (KA / nout).cpu().numpy()
        out["K_live"] = (KL / nout).cpu().numpy()
    with torch.no_grad():
        dead = [float(((torch.relu(z) > 0).sum(0) == 0).float().mean()) for z in zs[:-1]]
    return out, frac_below, dead


def hvp_top(iters=30):
    """Top Hessian eigenvalue of the training loss on the fixed 2048 examples (as rich.py): (largest algebraic,
    largest magnitude)."""
    loss = loss_fn(model(X_h), Y_h)
    g = torch.autograd.grad(loss, params, create_graph=True)

    def apply(v):
        return list(torch.autograd.grad(g, params, grad_outputs=v, retain_graph=True))

    def power(shift):
        gen = torch.Generator(device="cpu").manual_seed(0)
        v = [torch.randn(q.shape, generator=gen).to(dev) for q in params]
        lam = 0.0
        for _ in range(iters):
            nv = math.sqrt(sum((x * x).sum().item() for x in v))
            v = [x / nv for x in v]
            w = [a - shift * b for a, b in zip(apply(v), v)] if shift else apply(v)
            lam = sum((a * b).sum().item() for a, b in zip(v, w))
            v = [x.detach() for x in w]
        return lam

    lam_dom = power(0.0)
    if lam_dom >= 0:
        return lam_dom, lam_dom
    return power(lam_dom) + lam_dom, lam_dom


# ----------------------------------------------------------------------------- checkpoints
k_ep = sorted(set([0, 1, 2, 3, 5, 7] + list(range(10, args.epochs + 1, 10)) + [args.epochs]))
k_steps = [e * spe for e in k_ep]
h_ep = [0, 1, 3, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500]
h_steps = set(e * spe for e in h_ep if e <= args.epochs)
meta = dict(vars(args), name=name, steps_per_epoch=spe, total_steps=total_steps, params=sum(q.numel() for q in params),
            probe_ntk_train=probe_ntk_tr.tolist(), probe_ntk_test=probe_ntk_te.tolist(), ckpts=[], hessian=[],
            errors=[], diverged=False, finished=False)


def save_meta():
    tmp = os.path.join(rdir, "meta.json.tmp")
    json.dump(meta, open(tmp, "w"))
    os.replace(tmp, os.path.join(rdir, "meta.json"))


def checkpoint(step):
    t0 = time.time()
    ftr, fte = forward_all(X), forward_all(Xt)
    rec = dict(step=step, epoch=step / spe, train_loss=float(loss_fn(ftr, Y)),
               train_acc=float((ftr.argmax(1) == ytr_t).float().mean()), test_loss=float(loss_fn(fte, Yt)),
               test_acc=float((fte.argmax(1) == yte_t).float().mean()))
    try:
        ks, frac_below, dead = kernels(X_ntk)
        np.savez(os.path.join(rdir, f"ck_{step:07d}.npz"), **{k: v.astype(np.float32) for k, v in ks.items()})
        rec.update(frac_sqrtv_below_eps=frac_below, dead_frac=dead, kernels=sorted(ks))
    except Exception as e:  # keep training if a kernel computation fails
        meta["errors"].append([step, "kernel", repr(e)])
        log("kernel failed at step", step, repr(e))
    if step in h_steps:
        try:
            top, dom = hvp_top()
            meta["hessian"].append(dict(step=step, epoch=step / spe, top=top, dominant=dom))
        except Exception as e:
            meta["errors"].append([step, "hessian", repr(e)])
            log("hessian failed at step", step, repr(e))
    rec["seconds_ckpt"] = time.time() - t0
    rec["seconds_total"] = time.time() - t_start
    meta["ckpts"].append(rec)
    save_meta()


g = torch.Generator(device=dev).manual_seed(args.seed)  # data order, as rich.py with data_seed = seed
step = 0
epochs_run = args.epochs if args.max_epochs_run < 0 else min(args.epochs, args.max_epochs_run)
for ep in range(epochs_run):
    perm = torch.randperm(N, generator=g, device=dev)
    for i in range(0, N, args.batch):
        if step in k_steps:
            checkpoint(step)
        b = perm[i:i + args.batch]
        loss = loss_fn(model(X[b]), Y[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
    if not math.isfinite(float(loss.detach())):
        meta["diverged"] = True
        log("non-finite loss at step", step)
        break
    if (ep + 1) % 50 == 0:
        last = meta["ckpts"][-1]
        log(f"epoch {ep + 1} train_loss {last['train_loss']:.4f} test_acc {last['test_acc']:.4f} ({time.time() - t_start:.0f}s)")
if not meta["diverged"] and step in k_steps and meta["ckpts"][-1]["step"] != step:
    checkpoint(step)
meta["finished"] = epochs_run == args.epochs and not meta["diverged"]
meta["seconds"] = time.time() - t_start
meta["steps_run"] = step
save_meta()
open(os.path.join(args.out, name + ".done"), "w").write("ok\n")
log(f"done {step} steps ({time.time() - t_start:.0f}s)")
