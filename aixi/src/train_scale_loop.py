"""Exposure-threshold sweep for looped vs untied in-context learners (GPU version of train_scale.py).
Training data: the planted prior in law_data.npz (195 planted behaviours at fixed frequencies, as in Experiment 4).
Constant learning rate after warmup; an EMA of the weights is evaluated at each checkpoint:
excess log-loss over exact Bayes, and log2 q(b) for every behaviour over 8 input orders."""
import argparse, copy, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from common import Sampler, bayes_predict, NIN
from train_loop_model import LoopICL

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--loops", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--ckpts", default="1000,2000,4000,8000,16000")
ap.add_argument("--orders", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
LOG2 = math.log(2)
ckpts = [int(c) for c in a.ckpts.split(",")]

D = np.load("runs/exp4_scale/law_data.npz")
import pandas as pd
prior = pd.DataFrame({"tt": D["tt"], "p": D["p"] / D["p"].sum()})
tt_all = prior.tt.values.astype(np.int64)
N = len(tt_all)
ev = Sampler(prior, seed=12345)
exs, eys, _ = ev.batch(8192)
eb = np.clip(bayes_predict(prior, exs, eys), 1e-12, 1 - 1e-12)
eb_ll = np.where(eys == 1, np.log2(eb), np.log2(1 - eb))
exs_t, eys_t = torch.tensor(exs, device=dev), torch.tensor(eys, device=dev)
tr = Sampler(prior, seed=1000 + a.seed)

model = LoopICL(a.layers, a.d).to(dev)
ema = copy.deepcopy(model)
for p in ema.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
rng_e = np.random.default_rng(777)
eval_orders = [np.argsort(rng_e.random((N, NIN)), axis=1) for _ in range(a.orders)]


@torch.no_grad()
def evaluate(m):
    m.eval()
    ll = torch.cat([-F.binary_cross_entropy_with_logits(m(exs_t[i:i + 2048], eys_t[i:i + 2048], a.loops),
                                                         eys_t[i:i + 2048].float(), reduction="none") / LOG2
                    for i in range(0, 8192, 2048)]).cpu().numpy()
    excess = float((eb_ll - ll).mean())
    logq = np.zeros((N, a.orders))
    for r, order in enumerate(eval_orders):
        ys = (tt_all[:, None] >> order) & 1
        for i in range(0, N, 4096):
            xo = torch.tensor(order[i:i + 4096], device=dev); yo = torch.tensor(ys[i:i + 4096], device=dev)
            logq[i:i + 4096, r] = (-F.binary_cross_entropy_with_logits(m(xo, yo, a.loops), yo.float(), reduction="none")
                                   / LOG2).sum(1).cpu().numpy()
    m.train()
    return excess, logq


log, t0 = [], time.time()
for step in range(1, max(ckpts) + 1):
    xs, ys, _ = tr.batch(a.batch)
    xs, ys = torch.tensor(xs, device=dev), torch.tensor(ys, device=dev)
    loss = F.binary_cross_entropy_with_logits(model(xs, ys, a.loops), ys.float())
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
        print(f"[{a.out}] step {step} seqs {step * a.batch} excess {excess:.5f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"args": vars(a), "params": sum(p.numel() for p in model.parameters()), "log": log,
                   "compute_per_seq": a.layers * a.loops}, open(f"{a.out}.json", "w"))
print("done", a.out, flush=True)
