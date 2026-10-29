"""H4 (README hypothesis 1): Adam's sqrt(v_t) normalization acts as a negative feedback loop that makes it
implicitly minimize a generalization measure such as sharpness.

Predictions if true:
  (a) at matched training loss, Adam sits at lower sharpness (top Hessian eigenvalue, Hessian trace) than
      GD or GD with momentum;
  (b) the effect comes from v_t adapting: Adam with v_t frozen partway through ("adam_frozen", a fixed
      diagonal preconditioner) should lose it;
  (c) lower sharpness goes with better test performance.
Competing account (adaptive edge of stability, Cohen et al. 2022): the feedback holds the *preconditioned*
sharpness lambda_max(P^-1 H), P = diag(sqrt(v_hat) + eps), near (2 + 2 beta1) / ((1 - beta1) lr) = 38/lr,
and says nothing about raw sharpness.

MLP 64-128-128-10 (tanh) on sklearn digits, 600 train / 1197 test, full-batch MSE on one-hot targets
(MSE rather than cross-entropy, so curvature does not trivially vanish as the training loss goes to 0).

    python3 experiments/h4_adam_sharpness/sharpness.py [n_processes]
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS, MEASURE_EVERY, FREEZE_AT, N_TRAIN = 5000, 100, 300, 600
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
CONFIGS = [("gd", 0.05), ("gd", 0.1), ("gd", 0.2),
           ("momentum", 0.005), ("momentum", 0.01), ("momentum", 0.02),
           ("adam", 1e-3), ("adam", 1e-2), ("adam", 3e-2),
           ("adam_frozen", 1e-3), ("adam_frozen", 1e-2), ("adam_frozen", 3e-2)]
SEEDS = range(3)


def load_data(seed):
    digits = load_digits()
    x = torch.tensor(digits.data / 16.0, dtype=torch.float32)
    y = torch.tensor(digits.target)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    tr, te = perm[:N_TRAIN], perm[N_TRAIN:]
    return x[tr], y[tr], x[te], y[te]


def make_model():
    return torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Tanh(),
                               torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 10))


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def loss_fn(model, x, y):
    return F.mse_loss(model(x), F.one_hot(y, 10).float(), reduction="sum") / len(x)


def hvp_fn(model, x, y):
    """Returns v -> H v for the full-batch training loss, with v a flat vector."""
    params = list(model.parameters())
    grads = flat(torch.autograd.grad(loss_fn(model, x, y), params, create_graph=True))
    return lambda v: flat(torch.autograd.grad(grads @ v, params, retain_graph=True))


def top_eigenvalue(matvec, n, iters=30, seed=0):
    v = torch.randn(n, generator=torch.Generator().manual_seed(seed))
    v /= v.norm()
    for _ in range(iters):
        w = matvec(v)
        lam = float(v @ w)
        v = w / w.norm()
    return lam


def hessian_trace(hvp, n, samples=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    zs = [torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1 for _ in range(samples)]
    return float(np.mean([float(z @ hvp(z)) for z in zs]))


class Adam:
    """Plain Adam written out so the preconditioner P = sqrt(v_hat) + eps is visible and can be frozen."""

    def __init__(self, params, lr, freeze_at=None):
        self.params, self.lr, self.freeze_at, self.t = list(params), lr, freeze_at, 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.P = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        frozen = self.freeze_at is not None and self.t > self.freeze_at
        for p, m, v in zip(self.params, self.m, self.v):
            m.mul_(BETA1).add_(p.grad, alpha=1 - BETA1)
            if not frozen:
                v.mul_(BETA2).addcmul_(p.grad, p.grad, value=1 - BETA2)
        if not frozen:
            self.P = [(v / (1 - BETA2 ** self.t)).sqrt() + EPS for v in self.v]
        for p, m, P in zip(self.params, self.m, self.P):
            p -= self.lr * (m / (1 - BETA1 ** self.t)) / P


def run(cfg):
    opt_name, lr, seed = cfg
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x, y, x_test, y_test = load_data(seed)
    model = make_model()
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    if opt_name == "gd":
        opt = torch.optim.SGD(params, lr=lr)
    elif opt_name == "momentum":
        opt = torch.optim.SGD(params, lr=lr, momentum=BETA1)
    else:
        opt = Adam(params, lr, freeze_at=FREEZE_AT if opt_name == "adam_frozen" else None)

    # stability threshold: GD 2/lr, heavy-ball momentum (2+2b)/lr, Adam (on P^-1 H) (2+2b)/((1-b) lr)
    threshold = {"gd": 2 / lr, "momentum": (2 + 2 * BETA1) / lr}.get(opt_name, (2 + 2 * BETA1) / ((1 - BETA1) * lr))
    history, diverged_at = [], None
    for step in range(STEPS + 1):
        if step % MEASURE_EVERY == 0:
            hvp = hvp_fn(model, x, y)
            row = dict(step=step, sharpness=top_eigenvalue(hvp, n_params), trace=hessian_trace(hvp, n_params))
            if isinstance(opt, Adam) and opt.P is not None:
                p_inv_sqrt = 1 / flat(opt.P).sqrt()                       # symmetric form P^-1/2 H P^-1/2
                row["precond_sharpness"] = top_eigenvalue(lambda v: p_inv_sqrt * hvp(p_inv_sqrt * v), n_params)
            with torch.no_grad():
                row["train_loss"] = float(loss_fn(model, x, y))
                row["test_loss"] = float(loss_fn(model, x_test, y_test))
                row["test_acc"] = float((model(x_test).argmax(1) == y_test).float().mean())
            history.append(row)
        if step == STEPS:
            break
        loss = loss_fn(model, x, y)
        if not torch.isfinite(loss) or loss > 1e3:
            diverged_at = step
            break
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
    return dict(opt=opt_name, lr=lr, seed=seed, threshold=threshold, diverged_at=diverged_at, history=history)


if __name__ == "__main__":
    configs = [(o, lr, s) for o, lr in CONFIGS for s in SEEDS]
    results = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for r in pool.imap_unordered(run, configs):
            last = r["history"][-1]
            if r["diverged_at"] is not None:
                print(f"{r['opt']:12s} lr {r['lr']:<6g} seed {r['seed']}  diverged at step {r['diverged_at']}", flush=True)
                results.append(r)
                continue
            print(f"{r['opt']:12s} lr {r['lr']:<6g} seed {r['seed']}  train {last['train_loss']:.4f}  "
                  f"test acc {last['test_acc']:.3f}  sharpness {last['sharpness']:.1f}", flush=True)
            results.append(r)
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"))
