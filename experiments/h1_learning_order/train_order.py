"""H1 training: one in-context learner on the exact AIXI prior (same setup as aixi/ experiment 1).

At each checkpoint it records, for every behaviour, the per-prediction accuracy and the implied log2 q(b),
averaged over a fixed set of input orders.

    python3 experiments/h1_learning_order/train_order.py --d 64 --seed 0 --out experiments/h1_learning_order/runs/d64_s0
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from shared import PRIOR_PATH, eval_set, load_prior
from common import Sampler, ICL

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--orders", type=int, default=4)
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_num_threads(1)
torch.manual_seed(args.seed)

prior = load_prior(PRIOR_PATH)
n_beh = len(prior)
eval_xs, eval_ys = map(torch.tensor, eval_set(prior["tt"].values.astype(np.int64), args.orders))

model = ICL(layers=args.layers, d=args.d)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
warmup = 200
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
sampler = Sampler(prior, seed=1000 + args.seed)


@torch.no_grad()
def evaluate():
    """Per-behaviour accuracy and log2 q(b), each averaged over the input orders."""
    model.eval()
    logits = torch.cat([model(eval_xs[i:i + 4096], eval_ys[i:i + 4096]) for i in range(0, len(eval_xs), 4096)])
    model.train()
    y = eval_ys.float()
    acc = ((logits > 0).float() == y).float().mean(1)
    log2q = -F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(1) / math.log(2)
    return acc.view(n_beh, -1).mean(1).numpy(), log2q.view(n_beh, -1).mean(1).numpy()


checkpoints = sorted({0, 25, 50, 100, 200, *range(300, args.steps + 1, 150), args.steps})
accs, logqs = [], []
t0 = time.time()
for step in range(args.steps + 1):
    if step in checkpoints:
        acc, log2q = evaluate()
        accs.append(acc); logqs.append(log2q)
        print(f"[d{args.d} s{args.seed}] step {step:5d} mean acc {acc.mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    if step == args.steps:
        break
    xs, ys, _ = sampler.batch(args.batch)
    xs, ys = torch.tensor(xs), torch.tensor(ys)
    loss = F.binary_cross_entropy_with_logits(model(xs, ys), ys.float())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
np.savez(args.out + ".npz", steps=np.array(checkpoints), acc=np.array(accs), logq=np.array(logqs))
json.dump(vars(args) | {"params": sum(p.numel() for p in model.parameters()), "time": time.time() - t0},
          open(args.out + ".json", "w"))
