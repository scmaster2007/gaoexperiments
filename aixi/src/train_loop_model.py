"""The looped in-context learner used by train_loop.py and train_scale_loop.py."""
import torch
import torch.nn as nn
from common import Block, BITS


class LoopICL(nn.Module):
    """Same token layout as common.ICL. A stack of `layers` blocks is applied `loops` times,
    adding the input embedding before each pass. With loops=1 it is the untied model with `layers` blocks."""

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
