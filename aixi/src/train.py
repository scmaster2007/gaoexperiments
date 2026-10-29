"""Train one in-context learner on sequences sampled from the exact prior, then record:
  - excess log-loss over the exact Bayes predictor during training (bits per prediction),
  - the network's implied probability of every behaviour (chain rule over random input orders)."""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from common import load_prior, Sampler, bayes_predict, ICL, NIN

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out", default="run")
ap.add_argument("--prior", default="data/behaviours.csv")
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)

prior = load_prior(a.prior)
LOG2 = math.log(2)

# fixed held-out set from the prior, with exact Bayes predictions
ev = Sampler(prior, seed=12345)
exs, eys, _ = ev.batch(8192)
ebayes = np.clip(bayes_predict(prior, exs, eys), 1e-12, 1 - 1e-12)
ebayes_ll = np.where(eys == 1, np.log2(ebayes), np.log2(1 - ebayes))       # (N, 16) bits
exs_t, eys_t = torch.tensor(exs), torch.tensor(eys)

model = ICL(layers=a.layers, d=a.d)
nparams = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
warm = 500
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
tr = Sampler(prior, seed=1000 + a.seed)


def evaluate():
    model.eval()
    with torch.no_grad():
        ll = []
        for i in range(0, len(exs), 2048):
            lg = model(exs_t[i:i + 2048], eys_t[i:i + 2048])
            ll.append(-F.binary_cross_entropy_with_logits(lg, eys_t[i:i + 2048].float(), reduction="none") / LOG2)
        ll = torch.cat(ll).numpy()
    model.train()
    excess = ebayes_ll - ll                 # >= 0 on average; bits per prediction
    return float(excess.mean()), excess.mean(0).tolist(), float(-ll.mean())


curve = []
t0 = time.time()
for step in range(a.steps + 1):
    if step % 1000 == 0 or step == a.steps:
        ex, ex_pos, nll = evaluate()
        curve.append({"step": step, "excess_bits": ex, "nll_bits": nll, "time": time.time() - t0})
        print(f"[L{a.layers} s{a.seed}] step {step:6d}  excess {ex:.5f} bits/pred  nll {nll:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if step == a.steps:
        break
    xs, ys, _ = tr.batch(a.batch)
    xs, ys = torch.tensor(xs), torch.tensor(ys)
    loss = F.binary_cross_entropy_with_logits(model(xs, ys), ys.float())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

# implied joint probability of every behaviour, over R random input orders
R = 16
rng = np.random.default_rng(777)
tt = prior["tt"].values.astype(np.int64)
N = len(tt)
logq = np.zeros((N, R))
model.eval()
with torch.no_grad():
    for r in range(R):
        order = np.argsort(rng.random((N, NIN)), axis=1)
        ys = (tt[:, None] >> order) & 1
        for i in range(0, N, 4096):
            lg = model(torch.tensor(order[i:i + 4096]), torch.tensor(ys[i:i + 4096]))
            ll = -F.binary_cross_entropy_with_logits(lg, torch.tensor(ys[i:i + 4096]).float(), reduction="none") / LOG2
            logq[i:i + 4096, r] = ll.sum(1).numpy()

np.save(f"{a.out}_logq.npy", logq)
json.dump({"args": vars(a), "params": nparams, "curve": curve, "excess_by_position": ex_pos,
           "train_seconds": time.time() - t0}, open(f"{a.out}.json", "w"))
torch.save(model.state_dict(), f"{a.out}.pt")
print("done", a.out, f"{time.time()-t0:.0f}s", flush=True)
