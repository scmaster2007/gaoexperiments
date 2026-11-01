"""Realistic MLP training on CIFAR-10 with broad logging, for exploratory analysis.

Model: nn.Linear + ReLU, PyTorch default init (with biases). Squared loss on one-hot labels,
loss = 0.5 * mean over examples of the squared error summed over the 10 outputs. Full 50k/10k CIFAR-10.

Logged (all saved to <out>/<name>.npz plus a small <name>.json):
  every step        minibatch loss; global gradient norm every 10 steps
  ~70 checkpoints   train/test loss and accuracy; test outputs (fp16, all 10k); train outputs on a fixed
                    2k subset (fp16); correctness of all 50k training examples (packed bits)
  ~13 deep ckpts    per layer: Frobenius norm, max |w|, full singular values, distance from init (W and b);
                    per hidden neuron: squared incoming norm (weights + bias) and squared outgoing norm;
                    representations on 1000 fixed test inputs: dead-unit fraction, effective rank, linear CKA
                    with labels, mean norm; empirical NTK (output-averaged, with biases) on 500 fixed points
                    (250 train + 250 test); top Hessian eigenvalue of the training loss on 2048 fixed training
                    examples (largest algebraic and largest-magnitude); for Adam/AdamW also the same for P^-1 H with P = (1 - b1^t)(sqrt(vhat) + eps)
                    (preconditioned sharpness, Cohen et al. 2022) and second-moment statistics per layer
                    (quantiles of sqrt(vhat), fraction of coordinates with sqrt(vhat) < eps, 10 eps, 100 eps,
                    quantiles of |mhat| / (sqrt(vhat) + eps))
  weights           final weights (fp16) for every run; also at epoch 150 for seed 0
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
p.add_argument("--out", default=os.path.expanduser("~/rich"))
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--depth", type=int, default=4)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--opt", default="adam", choices=["adam", "adamw", "sgd"])
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eps", type=float, default=1e-8)
p.add_argument("--beta1", type=float, default=0.9)
p.add_argument("--beta2", type=float, default=0.999)
p.add_argument("--wd", type=float, default=0.01, help="AdamW decoupled weight decay")
p.add_argument("--momentum", type=float, default=0.9)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--epochs", type=int, default=500)
p.add_argument("--sched", default="const", choices=["const", "cosine"])
p.add_argument("--name", default="")
p.add_argument("--data_seed", type=int, default=-1, help="seed for the data order (default: same as --seed)")
p.add_argument("--small", action="store_true", help="CPU test mode: tiny subsets")
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)
name = args.name or (f"rich_{args.opt}_L{args.depth}_w{args.width}_b{args.batch}_lr{args.lr:g}"
                     f"_eps{args.eps:g}_{args.sched}{args.epochs}_s{args.seed}"
                     + (f"_ds{args.data_seed}" if args.data_seed >= 0 and args.data_seed != args.seed else ""))
t_start = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), name, *a, flush=True)


# ----------------------------------------------------------------------------- data
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
rng = np.random.default_rng(12345)  # fixed probe sets, identical for every run
probe_train_out = np.sort(rng.choice(N, min(2000, N), replace=False))
probe_rep = np.sort(rng.choice(len(Xt), min(1000, len(Xt)), replace=False))
probe_ntk_tr = np.sort(rng.choice(N, 250, replace=False))
probe_ntk_te = np.sort(rng.choice(len(Xt), 250, replace=False))
probe_hess = np.sort(rng.choice(N, min(2048, N), replace=False))
X_ntk = torch.cat([X[probe_ntk_tr], Xt[probe_ntk_te]])
X_rep, Y_rep = Xt[probe_rep], Yt[probe_rep]
X_h, Y_h = X[probe_hess], Y[probe_hess]

# ----------------------------------------------------------------------------- model / optimizer
model = core.realistic_mlp(3072, args.width, args.depth, seed=args.seed, device=dev)
linears = [m for m in model if isinstance(m, torch.nn.Linear)]
init = [(l.weight.detach().clone(), l.bias.detach().clone()) for l in linears]
params = list(model.parameters())
if args.opt == "adam":
    opt = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
elif args.opt == "adamw":
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.wd)
else:
    opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
spe = math.ceil(N / args.batch)
total_steps = spe * args.epochs


def lr_at(step):
    if args.sched == "const":
        return args.lr
    return 0.5 * args.lr * (1 + math.cos(math.pi * step / total_steps))


def forward_all(Xs, chunk=10000):
    with torch.no_grad():
        return torch.cat([model(Xs[i:i + chunk]) for i in range(0, len(Xs), chunk)])


def loss_fn(f, y):
    return 0.5 * ((f - y) ** 2).sum(1).mean()


# ----------------------------------------------------------------------------- checkpoints
ep_ck = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50] + list(range(60, args.epochs + 1, 10))
ep_ck = sorted(set(e for e in ep_ck if e <= args.epochs) | {args.epochs})
ckpt_steps = sorted(set([0, 1, 3, 10, 30, 100, 300]) | {e * spe for e in ep_ck})
ckpt_steps = [s for s in ckpt_steps if s <= total_steps]
deep_ep = [1, 3, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500]
deep_steps = sorted(set([0] + [e * spe for e in deep_ep if e <= args.epochs] + [total_steps]))
snap_steps = {total_steps} | ({150 * spe} if args.seed == 0 and 150 <= args.epochs else set())

rec = {k: [] for k in ("step", "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr")}
test_out, train_out, train_bits = [], [], []
deep = {"step": [], "lr": []}
deep_arrays = {}


def put(key, val):
    deep_arrays.setdefault(key, []).append(val)


def ntk_realistic(Xs):
    """Output-averaged empirical NTK of the nn.Sequential MLP (weights and biases), exact per-layer identity:
    sum_l <delta_l, delta_l'> (<h_{l-1}, h_{l-1}'> + 1)."""
    h = Xs
    zs, hs = [], [Xs]
    for l, lin in enumerate(linears):
        z = lin(h)
        if l < len(linears) - 1:
            zs.append(z)
            h = torch.relu(z)
            hs.append(h)
    f = z
    K = torch.zeros(len(Xs), len(Xs), dtype=torch.float64, device=dev)
    hL = hs[-1].detach().double()
    K += hL @ hL.T + 1.0
    G = [(hh.detach().double() @ hh.detach().double().T + 1.0) for hh in hs[:-1]]
    for k in range(f.shape[1]):
        grads = torch.autograd.grad(f[:, k].sum(), zs, retain_graph=(k < f.shape[1] - 1))
        for l, d in enumerate(grads):
            d = d.detach().double()
            K += (d @ d.T) * G[l] / f.shape[1]
    return K.float().cpu().numpy()


def hvp_top(precond=None, iters=30):
    """Eigenvalues of H (or of P^-1/2 H P^-1/2, which has the same spectrum as P^-1 H) for the loss on the fixed
    batch, by power iteration. Returns (largest algebraic eigenvalue, eigenvalue of largest magnitude). When the
    largest-magnitude one is negative, a second power iteration on (A - lam_dom I) gives the largest algebraic one."""
    loss = loss_fn(model(X_h), Y_h)
    g = torch.autograd.grad(loss, params, create_graph=True)
    s = None if precond is None else [pp.rsqrt() for pp in precond]

    def apply(v):
        u = v if s is None else [x * si for x, si in zip(v, s)]
        Hu = torch.autograd.grad(g, params, grad_outputs=u, retain_graph=True)
        return list(Hu) if s is None else [x * si for x, si in zip(Hu, s)]

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


def eff_rank(H):
    s = torch.linalg.svdvals(H.double())
    p_ = s ** 2 / (s ** 2).sum()
    p_ = p_[p_ > 0]
    return float(torch.exp(-(p_ * p_.log()).sum()))


def cka_linear(A, B):
    A = A.double() - A.double().mean(0)
    B = B.double() - B.double().mean(0)
    num = (A.T @ B).norm() ** 2
    return float(num / ((A.T @ A).norm() * (B.T @ B).norm() + 1e-300))


def sample_q(t, qs, k=1_000_000):
    t = t.reshape(-1)
    if t.numel() > k:
        idx = torch.randint(0, t.numel(), (k,), generator=torch.Generator(device="cpu").manual_seed(1)).to(t.device)
        t = t[idx]
    return torch.quantile(t.float(), torch.tensor(qs, device=t.device)).cpu().numpy()


QS = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


def do_deep(step):
    deep["step"].append(step)
    deep["lr"].append(lr_at(step))
    # weights
    fro, mx, dW, db, svs = [], [], [], [], []
    for l, lin in enumerate(linears):
        W, b = lin.weight.detach(), lin.bias.detach()
        fro.append(float(W.norm())); mx.append(float(W.abs().max()))
        dW.append(float((W - init[l][0]).norm())); db.append(float((b - init[l][1]).norm()))
        svs.append(torch.linalg.svdvals(W.double()).float().cpu().numpy())
    put("w_fro", np.array(fro)); put("w_maxabs", np.array(mx)); put("w_dist_init", np.array(dW))
    put("b_dist_init", np.array(db))
    for l, s in enumerate(svs):
        put(f"svals_layer{l}", s)
    # per-neuron balance
    ins, outs = [], []
    for l in range(len(linears) - 1):
        W, b = linears[l].weight.detach(), linears[l].bias.detach()
        Wn = linears[l + 1].weight.detach()
        ins.append(((W ** 2).sum(1) + b ** 2).cpu().numpy())
        outs.append((Wn ** 2).sum(0).cpu().numpy())
    put("neuron_in_sq", np.stack(ins)); put("neuron_out_sq", np.stack(outs))
    # representations
    with torch.no_grad():
        h = X_rep
        dead, er, cka, mn = [], [], [], []
        for l, lin in enumerate(linears):
            h = lin(h)
            if l < len(linears) - 1:
                h = torch.relu(h)
                dead.append(float(((h > 0).sum(0) == 0).float().mean()))
            er.append(eff_rank(h)); cka.append(cka_linear(h, Y_rep)); mn.append(float(h.norm(dim=1).mean()))
    put("rep_dead_frac", np.array(dead)); put("rep_eff_rank", np.array(er)); put("rep_cka_labels", np.array(cka))
    put("rep_mean_norm", np.array(mn))
    # NTK
    put("ntk", ntk_realistic(X_ntk))
    # sharpness
    put("sharpness", np.array(hvp_top()))
    if args.opt in ("adam", "adamw"):
        st = [opt.state[q] for q in params]
        if len(st[0]) == 0:
            put("precond_sharpness", np.array([np.nan, np.nan]))
        else:
            t = int(st[0]["step"])
            b1c, b2c = 1 - args.beta1 ** t, 1 - args.beta2 ** t
            P = [b1c * ((s["exp_avg_sq"] / b2c).sqrt() + args.eps) for s in st]
            put("precond_sharpness", np.array(hvp_top(precond=P)))
            vq, frac, rq = [], [], []
            for q, s in zip(params, st):
                sv = (s["exp_avg_sq"] / b2c).sqrt()
                vq.append(sample_q(sv, QS))
                frac.append([float((sv < args.eps * c).float().mean()) for c in (1, 10, 100)])
                rq.append(sample_q((s["exp_avg"] / b1c).abs() / (sv + args.eps), QS))
            put("adam_sqrtv_quantiles", np.stack(vq)); put("adam_frac_sqrtv_below_eps_x1_10_100", np.array(frac))
            put("adam_ratio_quantiles", np.stack(rq))


deep_errors = []


def safe_deep(step):
    try:
        do_deep(step)
    except Exception as e:  # e.g. non-finite weights
        deep_errors.append([step, repr(e)])
        log("deep diagnostics failed at step", step, repr(e))


def do_ckpt(step):
    ftr = forward_all(X)
    fte = forward_all(Xt)
    rec["step"].append(step); rec["epoch"].append(step / spe); rec["lr"].append(lr_at(step))
    rec["train_loss"].append(float(loss_fn(ftr, Y))); rec["train_acc"].append(float((ftr.argmax(1) == ytr_t).float().mean()))
    rec["test_loss"].append(float(loss_fn(fte, Yt))); rec["test_acc"].append(float((fte.argmax(1) == yte_t).float().mean()))
    test_out.append(fte.half().cpu().numpy())
    train_out.append(ftr[probe_train_out].half().cpu().numpy())
    train_bits.append(np.packbits((ftr.argmax(1) == ytr_t).cpu().numpy()))


snaps = {}
step_loss_buf = torch.zeros(total_steps, device=dev)
grad_norm_buf = torch.zeros(total_steps // 10 + 1, device=dev)
g = torch.Generator(device=dev).manual_seed(args.seed if args.data_seed < 0 else args.data_seed)
step = 0
diverged = False
for ep in range(args.epochs):
    perm = torch.randperm(N, generator=g, device=dev)
    for i in range(0, N, args.batch):
        if step in ckpt_steps:
            do_ckpt(step)
        if step in deep_steps:
            safe_deep(step)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        b = perm[i:i + args.batch]
        loss = loss_fn(model(X[b]), Y[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step % 10 == 0:
            grad_norm_buf[step // 10] = torch.stack([(q.grad ** 2).sum() for q in params]).sum().sqrt()
        opt.step()
        step_loss_buf[step] = loss.detach()
        step += 1
    if not math.isfinite(float(step_loss_buf[step - 1])):
        diverged = True
    if diverged:
        log("non-finite loss at step", step)
        break
    if (ep + 1) % 50 == 0:
        log(f"epoch {ep + 1} train_loss {rec['train_loss'][-1]:.4f} test_acc {rec['test_acc'][-1]:.4f} "
            f"({time.time() - t_start:.0f}s)")
    if step in snap_steps:
        snaps[step] = {f"w{l}": lin.weight.detach().half().cpu().numpy() for l, lin in enumerate(linears)}
        snaps[step].update({f"b{l}": lin.bias.detach().half().cpu().numpy() for l, lin in enumerate(linears)})
if not diverged:
    if step in ckpt_steps and (not rec["step"] or rec["step"][-1] != step):
        do_ckpt(step)
    if step in deep_steps and (not deep["step"] or deep["step"][-1] != step):
        safe_deep(step)

arrays = {
    "ckpt_step": np.array(rec["step"]), "test_out": np.stack(test_out), "train_out_probe": np.stack(train_out),
    "train_correct_bits": np.stack(train_bits), "step_loss": step_loss_buf[:step].cpu().numpy(),
    "grad_norm_every10": grad_norm_buf.cpu().numpy(),
    "deep_step": np.array(deep["step"]), "probe_train_out": probe_train_out, "probe_rep": probe_rep,
    "probe_ntk_train": probe_ntk_tr, "probe_ntk_test": probe_ntk_te, "probe_hess": probe_hess,
}
for k, v in deep_arrays.items():
    try:
        arrays["deep_" + k] = np.stack(v)
    except ValueError:
        for j, x in enumerate(v):
            arrays[f"deep_{k}_{j}"] = x
for s, d in snaps.items():
    for k, v in d.items():
        arrays[f"weights_step{s}_{k}"] = v
np.savez_compressed(os.path.join(args.out, name + ".npz"), **arrays)
meta = dict(vars(args), name=name, steps_per_epoch=spe, total_steps=total_steps, steps_run=step, diverged=diverged,
            seconds=time.time() - t_start, params=sum(q.numel() for q in params), curves=rec,
            deep_steps_done=deep["step"], deep_lr=deep["lr"], deep_errors=deep_errors,
            sharpness_top=[float(x[0]) for x in deep_arrays.get("sharpness", [])],
            sharpness_dominant=[float(x[1]) for x in deep_arrays.get("sharpness", [])],
            precond_sharpness_top=[float(x[0]) for x in deep_arrays.get("precond_sharpness", [])],
            precond_sharpness_dominant=[float(x[1]) for x in deep_arrays.get("precond_sharpness", [])])
json.dump(meta, open(os.path.join(args.out, name + ".json"), "w"))
log(f"done {step} steps, final train_loss {rec['train_loss'][-1]:.4f} test_acc {rec['test_acc'][-1]:.4f} "
    f"({time.time() - t_start:.0f}s)")
