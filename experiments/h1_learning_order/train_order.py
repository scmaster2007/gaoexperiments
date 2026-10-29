"""H1 (from "Let's agree to agree"): do networks learn behaviours in a shared order, and does an exact
complexity measure (shortest program length K) predict that order?

Trains one in-context learner on the exact prior (same setup as aixi/ experiment 1) and records, at
every checkpoint, each behaviour's per-prediction accuracy and implied log2 q(b) over a fixed set of
input orders. Run from the repo root."""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../aixi/src"))
from common import load_prior, Sampler, bayes_predict, ICL, NIN

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--orders", type=int, default=4)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)

prior = load_prior(os.path.join(os.path.dirname(__file__), "../../aixi/data/behaviours.csv"))
tt = prior["tt"].values.astype(np.int64)
N, R = len(tt), a.orders

# every behaviour shown in R fixed input orders (shared across runs)
rng = np.random.default_rng(777)
exs = np.argsort(rng.random((N * R, NIN)), axis=1)
eys = (np.repeat(tt, R)[:, None] >> exs) & 1
exs_t, eys_t = torch.tensor(exs), torch.tensor(eys)

model = ICL(layers=a.layers, d=a.d)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
warm = 200
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
tr = Sampler(prior, seed=1000 + a.seed)


def evaluate():
    model.eval()
    acc, lq = [], []
    with torch.no_grad():
        for i in range(0, len(exs), 4096):
            lg = model(exs_t[i:i + 4096], eys_t[i:i + 4096])
            y = eys_t[i:i + 4096].float()
            acc.append(((lg > 0).float() == y).float().mean(1))
            lq.append((-F.binary_cross_entropy_with_logits(lg, y, reduction="none")).sum(1) / math.log(2))
    model.train()
    return torch.cat(acc).view(N, R).mean(1).numpy(), torch.cat(lq).view(N, R).mean(1).numpy()


ckpts = sorted(set([0, 25, 50, 100, 200] + list(range(300, a.steps + 1, 150)) + [a.steps]))
ACC, LQ = [], []
t0 = time.time()
for step in range(a.steps + 1):
    if step in ckpts:
        acc, lq = evaluate()
        ACC.append(acc); LQ.append(lq)
        print(f"[d{a.d} s{a.seed}] step {step:5d} mean acc {acc.mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
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

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out + ".npz", steps=np.array(ckpts), acc=np.array(ACC), logq=np.array(LQ))
json.dump(vars(a) | {"params": sum(p.numel() for p in model.parameters()), "time": time.time() - t0},
          open(a.out + ".json", "w"))
