"""Looped (weight-tied) in-context learner vs untied baselines, on the exact-prior task.
A stack of `layers` blocks is applied `loops` times, adding the input embedding before each pass.
--train_loops: a fixed loop count, or 'rand:R' to draw the loop count uniformly from 1..R at every step.
After training, the model is evaluated at every loop count in --eval_loops (default: the training count):
excess log-loss over exact Bayes, and log2 q(b) for every behaviour over 16 input orders."""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import load_prior, Sampler, bayes_predict, Block, BITS, NIN

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--train_loops", default="1")
ap.add_argument("--eval_loops", default="")
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--prior", default="data/behaviours.csv")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
torch.manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
dev = a.device
LOG2 = math.log(2)


class LoopICL(nn.Module):
    """Same token layout as common.ICL. With loops=1 it is exactly the untied model with `layers` blocks."""

    def __init__(self, layers, d, heads=4):
        super().__init__()
        self.bitproj = nn.Linear(4, d)
        self.tokentype = nn.Embedding(2, d)
        self.yemb = nn.Embedding(2, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.register_buffer("bits", torch.tensor(2 * BITS - 1))

    def forward(self, xs, ys, loops=1):
        B, n = xs.shape
        xb = self.bitproj(self.bits[xs])
        e = torch.stack([xb + self.tokentype.weight[0], xb + self.tokentype.weight[1] + self.yemb(ys)], dim=2).reshape(B, 2 * n, -1)
        h = torch.zeros_like(e)
        for _ in range(loops):
            h = h + e
            for blk in self.blocks:
                h = blk(h)
        return self.head(self.lnf(h[:, 0::2])).squeeze(-1)


prior = load_prior(a.prior)
ev = Sampler(prior, seed=12345)
exs, eys, _ = ev.batch(8192)
eb = np.clip(bayes_predict(prior, exs, eys), 1e-12, 1 - 1e-12)
eb_ll = np.where(eys == 1, np.log2(eb), np.log2(1 - eb))
exs_t, eys_t = torch.tensor(exs, device=dev), torch.tensor(eys, device=dev)

if a.train_loops.startswith("rand:"):
    Rmax = int(a.train_loops.split(":")[1])
    lrng = np.random.default_rng(99 + a.seed)
    draw = lambda: int(lrng.integers(1, Rmax + 1))
    mean_loops = (Rmax + 1) / 2
    default_eval = [Rmax]
else:
    r0 = int(a.train_loops)
    draw = lambda: r0
    mean_loops = r0
    default_eval = [r0]
eval_loops = [int(x) for x in a.eval_loops.split(",")] if a.eval_loops else default_eval

model = LoopICL(a.layers, a.d).to(dev)
nparams = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
tr = Sampler(prior, seed=1000 + a.seed)


@torch.no_grad()
def excess(loops):
    model.eval()
    ll = torch.cat([-F.binary_cross_entropy_with_logits(model(exs_t[i:i + 2048], eys_t[i:i + 2048], loops),
                                                         eys_t[i:i + 2048].float(), reduction="none") / LOG2
                    for i in range(0, 8192, 2048)]).cpu().numpy()
    model.train()
    return float((eb_ll - ll).mean())


t0 = time.time()
for step in range(1, a.steps + 1):
    xs, ys, _ = tr.batch(a.batch)
    xs, ys = torch.tensor(xs, device=dev), torch.tensor(ys, device=dev)
    loss = F.binary_cross_entropy_with_logits(model(xs, ys, draw()), ys.float())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 2500 == 0:
        print(f"[{a.out}] step {step} excess@{eval_loops[0]} {excess(eval_loops[0]):.5f} ({time.time()-t0:.0f}s)", flush=True)

tt = prior["tt"].values.astype(np.int64)
N = len(tt)
rng = np.random.default_rng(777)
orders = [np.argsort(rng.random((N, NIN)), axis=1) for _ in range(16)]
res = {}
model.eval()
with torch.no_grad():
    for L in eval_loops:
        logq = np.zeros((N, len(orders)))
        for r, order in enumerate(orders):
            ys = (tt[:, None] >> order) & 1
            for i in range(0, N, 4096):
                xo = torch.tensor(order[i:i + 4096], device=dev); yo = torch.tensor(ys[i:i + 4096], device=dev)
                ll = -F.binary_cross_entropy_with_logits(model(xo, yo, L), yo.float(), reduction="none") / LOG2
                logq[i:i + 4096, r] = ll.sum(1).cpu().numpy()
        np.save(f"{a.out}_r{L}_logq.npy", logq)
        res[str(L)] = excess(L)
json.dump({"args": vars(a), "params": nparams, "excess": res,
           "compute": a.steps * a.batch * a.layers * mean_loops,  # block passes x sequences
           "train_seconds": time.time() - t0}, open(f"{a.out}.json", "w"))
print("done", a.out, res, f"{time.time()-t0:.0f}s", flush=True)
