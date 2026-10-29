"""Shared pieces: the exact prior over behaviours, the exact Bayes predictor, data sampling, the model."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

NIN = 16          # number of distinct inputs (4-bit)
BITS = np.array([[(x >> k) & 1 for k in range(4)] for x in range(NIN)], dtype=np.float32)


def load_prior(path="data/behaviours.csv", base=5.0, Lmax=13):
    """Behaviour table plus prior P(b) = sum over programs p computing b of base^-|p|, normalised."""
    d = pd.read_csv(path)
    w = base ** -np.arange(Lmax + 1)
    counts = d[[f"n{L}" for L in range(Lmax + 1)]].values
    p = (counts * w).sum(1)
    d["p"] = p / p.sum()
    d["log2p"] = np.log2(d["p"])
    return d.reset_index(drop=True)


class Sampler:
    def __init__(self, prior, seed):
        self.tt = prior["tt"].values.astype(np.int64)
        self.cdf = np.cumsum(prior["p"].values)
        self.cdf[-1] = 1.0
        self.rng = np.random.default_rng(seed)

    def batch(self, B):
        idx = np.searchsorted(self.cdf, self.rng.random(B), side="right")
        tt = self.tt[idx]
        xs = np.argsort(self.rng.random((B, NIN)), axis=1)          # random order of all 16 inputs
        ys = (tt[:, None] >> xs) & 1
        return xs, ys, idx


def bayes_predict(prior, xs, ys):
    """Exact posterior predictive P(y_i = 1 | (x_1,y_1)..(x_{i-1},y_{i-1}), x_i) under the prior."""
    tt = prior["tt"].values.astype(np.int64)
    p = prior["p"].values
    B, n = xs.shape
    mask = np.ones((B, len(tt)), dtype=bool)
    out = np.zeros((B, n))
    for i in range(n):
        bit = ((tt[None, :] >> xs[:, i:i + 1]) & 1).astype(bool)
        wm = mask * p[None, :]
        out[:, i] = (wm * bit).sum(1) / wm.sum(1)
        mask &= bit == ys[:, i:i + 1].astype(bool)
    return out


class Block(nn.Module):
    """Pre-LayerNorm transformer block with causal self-attention."""

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D)
        x = x + self.o(a)
        return x + self.mlp(self.ln2(x))


class ICL(nn.Module):
    """Tokens alternate: query q_i = x_i, then pair p_i = (x_i, y_i). Prediction for y_i is read at q_i.
    Inputs are embedded from their bits; there are no positional embeddings, so examples are exchangeable."""

    def __init__(self, layers=2, d=64, heads=4):
        super().__init__()
        self.bitproj = nn.Linear(4, d)
        self.tokentype = nn.Embedding(2, d)      # 0 = query, 1 = pair
        self.yemb = nn.Embedding(2, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.register_buffer("bits", torch.tensor(2 * BITS - 1))

    def forward(self, xs, ys):
        B, n = xs.shape
        xb = self.bitproj(self.bits[xs])                              # (B, n, d)
        q = xb + self.tokentype.weight[0]
        pr = xb + self.tokentype.weight[1] + self.yemb(ys)
        h = torch.stack([q, pr], dim=2).reshape(B, 2 * n, -1)         # q1 p1 q2 p2 ...
        L = 2 * n
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=xs.device), 1)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.lnf(h[:, 0::2])).squeeze(-1)           # logits at query positions, (B, n)
