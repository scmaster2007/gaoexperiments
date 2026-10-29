"""Exposure-threshold law in Pythia language models.
Each sequence: <eos> then 16 examples "b0 b1 b2 b3 = y ;" written as tokens " 0"/" 1"/" ="/" ;".
The model is trained (all weights) to predict y from the earlier examples, with a 2-way softmax over " 0"/" 1".
Training sequences are sampled from the planted prior p; the exact Bayes predictor is computed on the GPU.
At each checkpoint (and at step 0) an EMA copy of the weights is evaluated:
  excess log-loss over exact Bayes on 8192 held-out sequences, and log2 q(b) for every behaviour over 8 input orders."""
import argparse, copy, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM, AutoConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-70m")
ap.add_argument("--scratch", action="store_true", help="random init with the same architecture")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--ckpts", default="1000,2000,4000,8000,16000")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="runs/exp4_scale/law_data.npz")
ap.add_argument("--out", required=True)
ap.add_argument("--max_minutes", type=float, default=1e9, help="stop after the last checkpoint that fits")
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = "cuda"
NIN = 16
LOG2 = math.log(2)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

D = np.load(a.data)
TT_np, P_np = D["tt"], D["p"] / D["p"].sum()
N = len(TT_np)
TT = torch.tensor(TT_np, device=dev)
PT = torch.tensor(P_np, device=dev, dtype=torch.float64)
BT = torch.stack([((TT >> x) & 1).bool() for x in range(NIN)])           # (16, N)

tok = AutoTokenizer.from_pretrained(a.model)
ids = {s: tok.encode(s) for s in [" 0", " 1", " =", " ;"]}
assert all(len(v) == 1 for v in ids.values()), ids
T0, T1, TEQ, TSC = (ids[s][0] for s in [" 0", " 1", " =", " ;"])
EOS = tok.eos_token_id if tok.eos_token_id is not None else 0
BITS = torch.tensor([[(x >> k) & 1 for k in range(4)] for x in range(NIN)], device=dev)
EQPOS = torch.tensor([1 + 7 * i + 4 for i in range(NIN)], device=dev)     # position of " =" for example i


def build(xs, ys):
    """xs, ys: (B, 16) long tensors on the GPU -> input_ids (B, 113)."""
    B = xs.shape[0]
    bits = BITS[xs]                                                         # (B, 16, 4)
    btok = torch.where(bits == 1, T1, T0)
    ytok = torch.where(ys == 1, T1, T0)
    ex = torch.cat([btok, torch.full((B, NIN, 1), TEQ, device=dev), ytok[..., None],
                    torch.full((B, NIN, 1), TSC, device=dev)], dim=2).reshape(B, NIN * 7)
    return torch.cat([torch.full((B, 1), EOS, device=dev), ex], dim=1)


def logits01(model, xs, ys):
    """2-way logits (B, 16, 2) for y_i, read at the ' =' position (which predicts the next token)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=build(xs, ys)).logits                         # (B, L, V)
    h = out[:, EQPOS, :]
    return torch.stack([h[..., T0], h[..., T1]], dim=-1).float()


def ll_bits(model, xs, ys):
    lg = logits01(model, xs, ys)
    return torch.gather(F.log_softmax(lg, -1), 2, ys[..., None]).squeeze(-1) / LOG2   # (B, 16)


def bayes(xs, ys):
    B = xs.shape[0]
    Wm = PT.expand(B, -1).clone()
    out = torch.empty(B, NIN, device=dev, dtype=torch.float64)
    for i in range(NIN):
        bit = BT[xs[:, i]]
        out[:, i] = torch.where(bit, Wm, 0.0).sum(1) / Wm.sum(1)
        Wm = torch.where(bit == ys[:, i:i + 1].bool(), Wm, 0.0)
    return out


class Sampler:          # identical RNG usage to the CPU experiments
    def __init__(self, seed):
        self.cdf = np.cumsum(P_np); self.cdf[-1] = 1.0
        self.rng = np.random.default_rng(seed)

    def batch(self, B):
        idx = np.searchsorted(self.cdf, self.rng.random(B), side="right")
        xs = np.argsort(self.rng.random((B, NIN)), axis=1)
        ys = (TT_np[idx][:, None] >> xs) & 1
        return torch.tensor(xs, device=dev), torch.tensor(ys, device=dev)


ev_x, ev_y = Sampler(12345).batch(8192)
pb = bayes(ev_x, ev_y).clamp(1e-12, 1 - 1e-12)
ev_bayes = torch.where(ev_y == 1, torch.log2(pb), torch.log2(1 - pb)).float()
rng_e = np.random.default_rng(777)
eval_orders = [np.argsort(rng_e.random((N, NIN)), axis=1) for _ in range(8)]

if a.scratch:
    model = GPTNeoXForCausalLM(AutoConfig.from_pretrained(a.model)).to(dev)
else:
    model = GPTNeoXForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)
model.config.use_cache = False
nparams = sum(p.numel() for p in model.parameters())
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
tr = Sampler(1000 + a.seed)
ckpts = [int(c) for c in a.ckpts.split(",")]


@torch.no_grad()
def evaluate(m, step):
    m.eval()
    ll = torch.cat([ll_bits(m, ev_x[i:i + 512], ev_y[i:i + 512]) for i in range(0, 8192, 512)])
    excess = float((ev_bayes - ll).mean())
    logq = np.zeros((N, len(eval_orders)))
    for r, order in enumerate(eval_orders):
        xo = torch.tensor(order, device=dev)
        yo = (TT[:, None] >> xo) & 1
        for i in range(0, N, 512):
            logq[i:i + 512, r] = ll_bits(m, xo[i:i + 512], yo[i:i + 512]).sum(1).cpu().numpy()
    np.save(f"{a.out}_n{step * a.batch}_logq.npy", logq)
    return excess


log, t0 = [], time.time()
ex0 = evaluate(model, 0)
log.append({"step": 0, "sequences": 0, "excess_bits": ex0, "time": time.time() - t0})
print(f"[{a.out}] params {nparams}  step 0 excess {ex0:.4f}  ({time.time()-t0:.0f}s)", flush=True)
model.train()
t_train = time.time()
for step in range(1, max(ckpts) + 1):
    xs, ys = tr.batch(a.batch)
    lg = logits01(model, xs, ys)
    loss = F.cross_entropy(lg.reshape(-1, 2), ys.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    dec = min(0.998, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(dec).add_(pm.detach(), alpha=1 - dec)
    if step == 50:
        rate = (time.time() - t_train) / 50
        print(f"[{a.out}] {rate*1000:.0f} ms/step -> {rate*max(ckpts)/60:.1f} min for {max(ckpts)} steps", flush=True)
    if step in ckpts:
        ex = evaluate(ema, step)
        log.append({"step": step, "sequences": step * a.batch, "excess_bits": ex, "time": time.time() - t0})
        print(f"[{a.out}] step {step} seqs {step*a.batch} excess {ex:.5f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"args": vars(a), "params": nparams, "log": log}, open(f"{a.out}.json", "w"))
        nxt = [c for c in ckpts if c > step]
        if nxt and (time.time() - t0) / 60 * nxt[0] / step > a.max_minutes:
            print(f"[{a.out}] stopping early: next checkpoint would exceed {a.max_minutes} min", flush=True)
            break
json.dump({"args": vars(a), "params": nparams, "log": log}, open(f"{a.out}.json", "w"))
print("done", a.out, flush=True)
