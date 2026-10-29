"""Scaling sweep: one run per model width, trained on the planted prior with a constant learning rate after warmup.
At each checkpoint (number of training sequences seen), an exponential moving average of the weights is evaluated:
  - excess log-loss over the exact Bayes predictor on held-out sequences,
  - the implied log2-probability of every behaviour (chain rule over R random input orders)."""
import argparse, copy, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from common import load_prior, Sampler, bayes_predict, ICL, NIN

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--ckpts", default="1000,2000,4000,8000,16000")
ap.add_argument("--pvec", required=True)
ap.add_argument("--orders", type=int, default=8)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
LOG2 = math.log(2)
ckpts = [int(c) for c in a.ckpts.split(",")]

prior = load_prior()
pv = np.load(a.pvec)
prior["p"] = pv / pv.sum()
tt_all = prior["tt"].values.astype(np.int64)
N = len(tt_all)

ev = Sampler(prior, seed=12345)
exs, eys, _ = ev.batch(8192)
ebayes = np.clip(bayes_predict(prior, exs, eys), 1e-12, 1 - 1e-12)
ebayes_ll = np.where(eys == 1, np.log2(ebayes), np.log2(1 - ebayes))
exs_t, eys_t = torch.tensor(exs), torch.tensor(eys)
tr = Sampler(prior, seed=1000 + a.seed)

model = ICL(layers=a.layers, d=a.d)
ema = copy.deepcopy(model)
for p in ema.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
warm = 500
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

rng_e = np.random.default_rng(777)
eval_orders = [np.argsort(rng_e.random((N, NIN)), axis=1) for _ in range(a.orders)]


def joint_bits(m, xs, ys):
    lg = m(xs, ys)
    return (-F.binary_cross_entropy_with_logits(lg, ys.float(), reduction="none") / LOG2).sum(1)


def evaluate(m):
    m.eval()
    with torch.no_grad():
        ll = []
        for i in range(0, len(exs), 2048):
            lg = m(exs_t[i:i + 2048], eys_t[i:i + 2048])
            ll.append(-F.binary_cross_entropy_with_logits(lg, eys_t[i:i + 2048].float(), reduction="none") / LOG2)
        excess = float((ebayes_ll - torch.cat(ll).numpy()).mean())
        logq = np.zeros((N, a.orders))
        for r, order in enumerate(eval_orders):
            ys = (tt_all[:, None] >> order) & 1
            for i in range(0, N, 4096):
                logq[i:i + 4096, r] = joint_bits(m, torch.tensor(order[i:i + 4096]), torch.tensor(ys[i:i + 4096])).numpy()
    m.train()
    return excess, logq


log, t0 = [], time.time()
for step in range(1, max(ckpts) + 1):
    xs, ys, _ = tr.batch(a.batch)
    xs, ys = torch.tensor(xs), torch.tensor(ys)
    loss = F.binary_cross_entropy_with_logits(model(xs, ys), ys.float())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    decay = min(0.998, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(pm, alpha=1 - decay)
    if step in ckpts:
        excess, logq = evaluate(ema)
        np.save(f"{a.out}_n{step * a.batch}_logq.npy", logq)
        log.append({"step": step, "sequences": step * a.batch, "excess_bits": excess, "time": time.time() - t0})
        print(f"[d{a.d}] step {step:6d}  sequences {step * a.batch:8d}  excess {excess:.5f} bits/pred  ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"args": vars(a), "params": sum(p.numel() for p in model.parameters()), "log": log},
                  open(f"{a.out}.json", "w"))
print("done", a.out, flush=True)
