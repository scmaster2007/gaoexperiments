"""Training-procedure variants that keep the population optimum (the exact Bayes predictor) unchanged:
  --target soft        train on the exact posterior predictive instead of sampled labels (oracle, lower variance)
  --is_tau T           sample behaviours from P^T, reweight each sequence by P/Q (self-normalised in the batch)
  --exch LAMBDA        penalise order-dependence of the joint log-probability (label-free coherence constraint)
  --opt sam            sharpness-aware minimisation on top of AdamW
Records the same quantities as train.py (excess loss vs exact Bayes; implied log-probability of every behaviour)."""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from common import load_prior, Sampler, bayes_predict, ICL, NIN

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--prior", default="data/behaviours.csv")
ap.add_argument("--target", choices=["hard", "soft"], default="hard")
ap.add_argument("--is_tau", type=float, default=1.0)
ap.add_argument("--exch", type=float, default=0.0)
ap.add_argument("--opt", choices=["adamw", "sam"], default="adamw")
ap.add_argument("--sam_rho", type=float, default=0.05)
ap.add_argument("--out", required=True)
ap.add_argument("--pvec", default=None, help=".npy prior vector replacing P (same behaviour order)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
LOG2 = math.log(2)

prior = load_prior(a.prior)
if a.pvec:
    pv = np.load(a.pvec)
    prior["p"] = pv / pv.sum()
    with np.errstate(divide="ignore"):
        prior["log2p"] = np.log2(prior["p"])
P = prior["p"].values
TT = torch.tensor(prior["tt"].values.astype(np.int64))
PT = torch.tensor(P, dtype=torch.float64)

# held-out evaluation set from the true prior, exact Bayes predictions (identical to train.py)
ev = Sampler(prior, seed=12345)
exs, eys, _ = ev.batch(8192)
ebayes = np.clip(bayes_predict(prior, exs, eys), 1e-12, 1 - 1e-12)
ebayes_ll = np.where(eys == 1, np.log2(ebayes), np.log2(1 - ebayes))
exs_t, eys_t = torch.tensor(exs), torch.tensor(eys)

# training sampler: proposal Q = P^tau (tau = 1 is the plain prior)
rng = np.random.default_rng(1000 + a.seed)
Q = P ** a.is_tau
Q /= Q.sum()
cdfQ = np.cumsum(Q); cdfQ[-1] = 1.0
with np.errstate(divide="ignore", invalid="ignore"):
    logw = np.log(P) - np.log(Q)                       # importance weight P/Q per behaviour


def sample(B):
    idx = np.searchsorted(cdfQ, rng.random(B), side="right")
    tt = prior["tt"].values.astype(np.int64)[idx]
    xs = np.argsort(rng.random((B, NIN)), axis=1)
    ys = (tt[:, None] >> xs) & 1
    return torch.tensor(xs), torch.tensor(ys), idx


BT = torch.stack([((TT >> x) & 1).bool() for x in range(NIN)])      # (16 inputs, N behaviours)
PT32 = PT.float()


def bayes_torch(xs, ys):
    """Exact posterior predictive for a batch: Wm holds the prior mass of behaviours still consistent."""
    B = xs.shape[0]
    Wm = PT32.expand(B, -1).clone()
    out = torch.empty(B, NIN)
    for i in range(NIN):
        bit = BT[xs[:, i]]                                   # (B, N)
        den = Wm.sum(1)
        out[:, i] = torch.where(bit, Wm, 0.0).sum(1) / den
        Wm = torch.where(bit == ys[:, i:i + 1].bool(), Wm, 0.0)
    return out


def joint_bits(logits, ys):
    return (-F.binary_cross_entropy_with_logits(logits, ys.float(), reduction="none") / LOG2).sum(1)


def loss_fn(xs, ys, idx, soft):
    logits = model(xs, ys)
    tgt = soft if soft is not None else ys.float()
    per_seq = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").mean(1)
    if a.is_tau != 1.0:
        w = torch.tensor(np.exp(logw[idx] - logw[idx].max()), dtype=torch.float32)
        loss = (w * per_seq).sum() / w.sum()
    else:
        loss = per_seq.mean()
    pen = torch.zeros(())
    if a.exch > 0:
        # (1) same pairs as the first half of the batch, in a different order
        h = xs.shape[0] // 2
        perm = torch.argsort(torch.rand(h, NIN), dim=1)
        xs2, ys2 = torch.gather(xs[:h], 1, perm), torch.gather(ys[:h], 1, perm)
        j1 = joint_bits(logits[:h], ys[:h])
        j2 = joint_bits(model(xs2, ys2), ys2)
        # (2) uniformly random truth tables (no labels or prior needed), two orders each
        r = 32
        tt_r = torch.randint(0, 2 ** NIN, (r,))
        oa = torch.argsort(torch.rand(r, NIN), dim=1); ob = torch.argsort(torch.rand(r, NIN), dim=1)
        ya, yb = (tt_r[:, None] >> oa) & 1, (tt_r[:, None] >> ob) & 1
        ja, jb = joint_bits(model(oa, ya), ya), joint_bits(model(ob, yb), yb)
        pen = 0.5 * ((j1 - j2) ** 2).mean() + 0.5 * ((ja - jb) ** 2).mean()
        loss = loss + a.exch * pen
    return loss, pen


model = ICL(layers=a.layers, d=a.d)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
warm = 500
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))


def evaluate():
    model.eval()
    with torch.no_grad():
        ll = []
        for i in range(0, len(exs), 2048):
            lg = model(exs_t[i:i + 2048], eys_t[i:i + 2048])
            ll.append(-F.binary_cross_entropy_with_logits(lg, eys_t[i:i + 2048].float(), reduction="none") / LOG2)
        ll = torch.cat(ll).numpy()
    model.train()
    ex = ebayes_ll - ll
    return float(ex.mean()), ex.mean(0).tolist()


curve, t0, pen_hist = [], time.time(), []
for step in range(a.steps + 1):
    if step % 1000 == 0 or step == a.steps:
        exb, ex_pos = evaluate()
        pm = float(np.mean(pen_hist)) if pen_hist else 0.0
        pen_hist = []
        curve.append({"step": step, "excess_bits": exb, "exch_penalty": pm, "time": time.time() - t0})
        print(f"[{a.out}] step {step:6d}  excess {exb:.5f} bits/pred  exch-pen {pm:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if step == a.steps:
        break
    xs, ys, idx = sample(a.batch)
    soft = bayes_torch(xs, ys) if a.target == "soft" else None
    if a.opt == "sam":
        loss, pen = loss_fn(xs, ys, idx, soft)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for p in model.parameters() if p.grad is not None]
        gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in params)) + 1e-12
        eps = []
        with torch.no_grad():
            for p in params:
                e = a.sam_rho * p.grad / gnorm
                p.add_(e); eps.append(e)
        loss2, _ = loss_fn(xs, ys, idx, soft)
        opt.zero_grad(set_to_none=True)
        loss2.backward()
        with torch.no_grad():
            for p, e in zip(params, eps):
                p.sub_(e)
    else:
        loss, pen = loss_fn(xs, ys, idx, soft)
        opt.zero_grad(set_to_none=True)
        loss.backward()
    pen_hist.append(float(pen))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

# implied joint probability of every behaviour over 16 random orders (same orders as train.py)
R = 16
rng_e = np.random.default_rng(777)
tt = prior["tt"].values.astype(np.int64)
N = len(tt)
logq = np.zeros((N, R))
model.eval()
with torch.no_grad():
    for r in range(R):
        order = np.argsort(rng_e.random((N, NIN)), axis=1)
        ys = (tt[:, None] >> order) & 1
        for i in range(0, N, 4096):
            lg = model(torch.tensor(order[i:i + 4096]), torch.tensor(ys[i:i + 4096]))
            logq[i:i + 4096, r] = joint_bits(lg, torch.tensor(ys[i:i + 4096])).numpy()

np.save(f"{a.out}_logq.npy", logq)
json.dump({"args": vars(a), "params": sum(p.numel() for p in model.parameters()), "curve": curve,
           "excess_by_position": ex_pos, "train_seconds": time.time() - t0}, open(f"{a.out}.json", "w"))
print("done", a.out, f"{time.time()-t0:.0f}s", flush=True)
