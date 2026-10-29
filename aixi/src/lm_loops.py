"""Zero-shot in-context prior of pretrained language models as a function of the number of recurrent steps.
Task format as in lm_law.py: <bos> then 16 examples "b0 b1 b2 b3 = y ;", scored with a 2-way softmax over the 0/1 tokens.
For each step count r:
  - excess log-loss over the exact Bayes predictor at each of the 16 context positions (held-out sequences
    sampled from the natural prior), and
  - log2 q(b) for every behaviour over several input orders.
kind = huginn: Huginn-0125 recurrent-depth model; all step counts come from one pass (coda applied at each checkpoint).
kind = ouro:   Ouro looped model; all step counts come from one pass (hidden state after each loop).
kind = plain:  an ordinary causal LM (one 'step')."""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import load_prior, Sampler, bayes_predict, NIN

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--kind", choices=["huginn", "ouro", "plain"], required=True)
ap.add_argument("--steps", default="1")
ap.add_argument("--orders", type=int, default=4)
ap.add_argument("--nheld", type=int, default=2048)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--out", required=True)
ap.add_argument("--check", action="store_true", help="huginn: compare the one-pass logits with model(num_steps=r)")
a = ap.parse_args()
dev = "cuda"
LOG2 = math.log(2)
steps = [int(s) for s in a.steps.split(",")]
if a.kind == "plain":
    steps = [1]

prior = load_prior("data/behaviours.csv")
TT = prior.tt.values.astype(np.int64)
N = len(TT)

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
fmt = None
for cand in [(" 0", " 1", " =", " ;"), ("0", "1", "=", ";")]:
    ids = [tok.encode(s, add_special_tokens=False) for s in cand]
    if all(len(i) == 1 for i in ids):
        fmt, (T0, T1, TEQ, TSC) = cand, [i[0] for i in ids]
        break
assert fmt is not None, "no single-token format"
START = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
BITS = torch.tensor([[(x >> k) & 1 for k in range(4)] for x in range(NIN)], device=dev)
EQPOS = torch.tensor([1 + 7 * i + 4 for i in range(NIN)], device=dev)


def build(xs, ys):
    B = xs.shape[0]
    btok = torch.where(BITS[xs] == 1, T1, T0)
    ytok = torch.where(ys == 1, T1, T0)
    ex = torch.cat([btok, torch.full((B, NIN, 1), TEQ, device=dev), ytok[..., None],
                    torch.full((B, NIN, 1), TSC, device=dev)], dim=2).reshape(B, NIN * 7)
    return torch.cat([torch.full((B, 1), START, device=dev), ex], dim=1)


model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
if a.kind == "ouro":
    model.model.total_ut_steps = max(steps)
    model.config.total_ut_steps = max(steps)
nparams = sum(p.numel() for p in model.parameters())


def two_way(lg, ys):
    l2 = torch.stack([lg[..., T0], lg[..., T1]], dim=-1).float()
    return torch.gather(F.log_softmax(l2, -1), 2, ys[..., None]).squeeze(-1) / LOG2      # (B, 16) bits


@torch.no_grad()
def huginn_multi(ids):
    m = model
    fc = m.freqs_cis[:, : ids.shape[1]]
    e = m.transformer.wte(ids)
    if m.emb_scale != 1:
        e = e * m.emb_scale
    bi = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
    for blk in m.transformer.prelude:
        bi += 1
        e = blk(e, fc, bi, None, None)
    x = m.initialize_state(e, scale=1.0)
    out = {}
    for s in range(1, max(steps) + 1):
        x, bi = m.core_block_forward(x, e, fc, None, None, bi, s - 1)
        if s in steps:
            z = m.transformer.ln_f(x)
            ci = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
            for blk in m.transformer.coda:
                ci -= 1
                z = blk(z, fc, ci, None, None)
            z = m.transformer.ln_f(z)
            out[s] = m.lm_head(z[:, EQPOS]).float()
    return out


@torch.no_grad()
def ll_all(xs, ys):
    ids = build(xs, ys)
    if a.kind == "huginn":
        torch.manual_seed(0)                       # Huginn's initial latent state is random
        lgs = huginn_multi(ids)
    elif a.kind == "ouro":
        _, hs, _ = model.model(input_ids=ids, use_cache=False)
        lgs = {r: model.lm_head(hs[r - 1][:, EQPOS]).float() for r in steps}
    else:
        lgs = {1: model(input_ids=ids).logits[:, EQPOS].float()}
    return {r: two_way(lg, ys) for r, lg in lgs.items()}


if a.kind == "huginn" and a.check:
    xs, ys, _ = Sampler(prior, seed=5).batch(8)
    xs, ys = torch.tensor(xs, device=dev), torch.tensor(ys, device=dev)
    mine = ll_all(xs, ys)
    for r in steps[:3]:
        torch.manual_seed(0)
        ref = two_way(model(input_ids=build(xs, ys), num_steps=r).logits[:, EQPOS], ys)
        print(f"check r={r}: max |diff| = {float((ref - mine[r]).abs().max()):.4f} bits", flush=True)

t0 = time.time()
# held-out sequences from the natural prior: excess over exact Bayes per position
hx, hy, _ = Sampler(prior, seed=12345).batch(a.nheld)
pb = np.clip(bayes_predict(prior, hx, hy), 1e-12, 1 - 1e-12)
bayes_ll = np.where(hy == 1, np.log2(pb), np.log2(1 - pb))
hx_t, hy_t = torch.tensor(hx, device=dev), torch.tensor(hy, device=dev)
held = {r: [] for r in steps}
for i in range(0, a.nheld, a.batch):
    for r, v in ll_all(hx_t[i:i + a.batch], hy_t[i:i + a.batch]).items():
        held[r].append(v.cpu().numpy())
excess_pos = np.stack([(bayes_ll - np.concatenate(held[r])).mean(0) for r in steps])          # (S, 16)
print(f"[{a.out}] held-out done ({time.time()-t0:.0f}s): " +
      "  ".join(f"r={r}: {excess_pos[k].mean():.4f}" for k, r in enumerate(steps)), flush=True)

# implied log2 q(b) for every behaviour
rng = np.random.default_rng(777)
orders = [np.argsort(rng.random((N, NIN)), axis=1) for _ in range(a.orders)]
logq = np.zeros((len(steps), N, a.orders))
for o, order in enumerate(orders):
    xo_all = torch.tensor(order, device=dev)
    yo_all = (torch.tensor(TT, device=dev)[:, None] >> xo_all) & 1
    for i in range(0, N, a.batch):
        for r, v in ll_all(xo_all[i:i + a.batch], yo_all[i:i + a.batch]).items():
            logq[steps.index(r), i:i + a.batch, o] = v.sum(1).cpu().numpy()
    print(f"[{a.out}] order {o} done ({time.time()-t0:.0f}s)", flush=True)
np.savez(f"{a.out}.npz", steps=np.array(steps), excess_pos=excess_pos, logq=logq)
json.dump({"args": vars(a), "params": nparams, "format": fmt, "start_token": START, "steps": steps,
           "excess_mean": excess_pos.mean(1).tolist(), "seconds": time.time() - t0}, open(f"{a.out}.json", "w"))
print("done", a.out, f"{time.time()-t0:.0f}s", flush=True)
