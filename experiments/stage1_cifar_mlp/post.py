"""Post-training measurements on saved final weights (rich.py outputs).
usage: python post.py RICH_DIR OUT_JSON
  1. loss/accuracy along straight lines in weight space (21 points) between: two seeds of the same config;
     Adam and SGD solutions from the same initialization; initialization and the final weights;
     the epoch-150 snapshot and the final weights (seed 0 runs)
  2. top-10 Hessian eigenvalues of the training loss (4096 fixed training examples) at the final weights, seed 0
  3. per-layer distances between final weights (seed to seed; Adam to SGD from the same init)"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

R, OUT = sys.argv[1], sys.argv[2]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
xtr, ytr, xte, yte = core.load_cifar(os.path.expanduser(os.environ.get("CIFAR", "~/data")))
small = bool(os.environ.get("SMALL"))
if small:
    xtr, ytr, xte, yte = xtr[:2000], ytr[:2000], xte[:500], yte[:500]
rng = np.random.default_rng(7)
itr = rng.choice(len(xtr), min(10000, len(xtr)), replace=False)
Xtr = torch.tensor(xtr[itr], device=dev); Ytr = torch.tensor(core.onehot(ytr[itr]), device=dev)
Xte = torch.tensor(xte, device=dev); Yte = torch.tensor(core.onehot(yte), device=dev)
ih = rng.choice(len(xtr), min(4096, len(xtr)), replace=False)
Xh = torch.tensor(xtr[ih], device=dev); Yh = torch.tensor(core.onehot(ytr[ih]), device=dev)


def load_final(npz, step=None):
    z = np.load(npz)
    keys = [k for k in z.files if k.startswith("weights_step")]
    steps = sorted({int(k.split("_")[1][4:]) for k in keys})
    s = steps[-1] if step is None else step
    L = len([k for k in keys if k.startswith(f"weights_step{s}_w")])
    return [(torch.tensor(z[f"weights_step{s}_w{l}"].astype(np.float32), device=dev),
             torch.tensor(z[f"weights_step{s}_b{l}"].astype(np.float32), device=dev)) for l in range(L)], steps


def init_weights(depth, width, seed):
    m = core.realistic_mlp(3072, width, depth, seed=seed, device=dev)
    return [(l.weight.detach(), l.bias.detach()) for l in m if isinstance(l, torch.nn.Linear)]


def fwd(W, X):
    h = X
    for l, (w, b) in enumerate(W):
        h = h @ w.T + b
        if l < len(W) - 1:
            h = torch.relu(h)
    return h


def evalw(W):
    with torch.no_grad():
        ftr = torch.cat([fwd(W, Xtr[i:i + 5000]) for i in range(0, len(Xtr), 5000)])
        fte = fwd(W, Xte)
    return (float(0.5 * ((ftr - Ytr) ** 2).sum(1).mean()), float((ftr.argmax(1) == Ytr.argmax(1)).float().mean()),
            float(0.5 * ((fte - Yte) ** 2).sum(1).mean()), float((fte.argmax(1) == Yte.argmax(1)).float().mean()))


def path(A, B, n=21):
    out = []
    for a in np.linspace(0, 1, n):
        W = [((1 - a) * wa + a * wb, (1 - a) * ba + a * bb) for (wa, ba), (wb, bb) in zip(A, B)]
        out.append(evalw(W))
    arr = np.array(out)
    return {"alpha": np.linspace(0, 1, n).round(3).tolist(), "train_loss": arr[:, 0].round(5).tolist(),
            "train_acc": arr[:, 1].round(4).tolist(), "test_loss": arr[:, 2].round(5).tolist(),
            "test_acc": arr[:, 3].round(4).tolist(),
            "train_loss_barrier": float(arr[:, 0].max() - max(arr[0, 0], arr[-1, 0])),
            "test_acc_dip": float(min(arr[0, 3], arr[-1, 3]) - arr[:, 3].min())}


def dists(A, B):
    return [float(torch.sqrt(((wa - wb) ** 2).sum() + ((ba - bb) ** 2).sum())) for (wa, ba), (wb, bb) in zip(A, B)]


def hess_top(W, k=10):
    import scipy.sparse.linalg as sla
    params = [t.clone().requires_grad_(True) for pair in W for t in pair]
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]

    def loss_fn():
        Wl = [(params[2 * i], params[2 * i + 1]) for i in range(len(W))]
        f = fwd(Wl, Xh)
        return 0.5 * ((f - Yh) ** 2).sum(1).mean()

    g = torch.autograd.grad(loss_fn(), params, create_graph=True)

    def mv(v):
        v = torch.tensor(np.asarray(v).ravel(), dtype=torch.float32, device=dev)
        vs = [c.view(s) for c, s in zip(torch.split(v, sizes), shapes)]
        hv = torch.autograd.grad(g, params, grad_outputs=vs, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in hv]).double().cpu().numpy()

    n = sum(sizes)
    op = sla.LinearOperator((n, n), matvec=mv, dtype=np.float64)
    vals = sla.eigsh(op, k=k, which="LA", tol=1e-3, maxiter=300, return_eigenvectors=False)
    return sorted(vals.tolist(), reverse=True)


files = sorted(glob.glob(os.path.join(R, "rich_*.npz")))
meta = {f: json.load(open(f[:-4] + ".json")) for f in files}
res = {"seed_pairs": {}, "adam_vs_sgd_same_init": {}, "init_to_final": {}, "epoch150_to_final": {},
       "hessian_top10": {}, "notes": "loss = 0.5 * mean squared error summed over outputs; train = 10k fixed subset"}


def cfg(m):
    return f"{m['opt']}_L{m['depth']}_w{m['width']}_b{m['batch']}_lr{m['lr']:g}_eps{m['eps']:g}_{m['sched']}{m['epochs']}"


by = {}
for f, m in meta.items():
    if m.get("data_seed", -1) not in (-1, m["seed"]) or m.get("diverged"):
        continue
    by.setdefault(cfg(m), {})[m["seed"]] = f
for c, seeds in sorted(by.items()):
    if 0 in seeds:
        W0, steps = load_final(seeds[0])
        m0 = meta[seeds[0]]
        Wi = init_weights(m0["depth"], m0["width"], 0)
        res["init_to_final"][c] = path(Wi, W0, 11)
        res["init_to_final"][c]["dist_by_layer"] = dists(Wi, W0)
        if len(steps) > 1:
            W150, _ = load_final(seeds[0], steps[0])
            res["epoch150_to_final"][c] = path(W150, W0, 11)
        if c.startswith(("adam_", "sgd_")) and m0["width"] == 1024 and m0["batch"] == 128:
            try:
                res["hessian_top10"][c] = hess_top(W0)
            except Exception as e:
                res["hessian_top10"][c] = repr(e)
        for s in (1, 2):
            if s in seeds:
                Ws, _ = load_final(seeds[s])
                r = path(W0, Ws); r["dist_by_layer"] = dists(W0, Ws)
                res["seed_pairs"][f"{c}__s0_s{s}"] = r
    print("done", c, flush=True)
    json.dump(res, open(OUT, "w"))
for L in (2, 4, 8, 16):
    a = by.get(f"adam_L{L}_w1024_b128_lr0.001_eps1e-08_const500", {})
    for lr in (0.01, 0.05):
        sg = by.get(f"sgd_L{L}_w1024_b128_lr{lr:g}_eps1e-08_const500", {})
        for s in (0, 1, 2):
            if s in a and s in sg:
                A, _ = load_final(a[s]); B, _ = load_final(sg[s])
                r = path(A, B); r["dist_by_layer"] = dists(A, B)
                res["adam_vs_sgd_same_init"][f"L{L}_sgdlr{lr:g}_s{s}"] = r
    json.dump(res, open(OUT, "w"))
print("all done", flush=True)
