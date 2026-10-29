"""H3 training (from "SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics"):
does Adam learn a staircase in the same order as SGD, and does its learning time scale with input
dimension d the way leap complexity predicts for SGD?

Targets on the hypercube {-1,1}^d:
    leap1: x1 + x1x2 + x1x2x3     staircase, leap 1   (SGD theory: ~d samples)
    leap3: x1x2x3                 single parity, leap 3 (SGD theory: ~d^2 samples)

2-layer ReLU net (width 256), online MSE on fresh batches of 64. For each monomial we record the first
step at which the network's projection onto it, E[f(x) m(x)], reaches 0.5 (its coefficient is 1).

    python3 experiments/h3_leap_adam/leap.py [n_processes]
"""
import json, os, sys
from itertools import product
from multiprocessing import Pool
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = {"leap1": [(0,), (0, 1), (0, 1, 2)], "leap3": [(0, 1, 2)]}
LRS = {"sgd": [0.05, 0.1], "adam": [1e-3, 3e-3]}
DIMS, SEEDS = [10, 20, 40, 80], range(4)
MAX_STEPS, EVAL_EVERY, BATCH, WIDTH = 60000, 10, 64, 256


def hypercube(n, d, gen):
    return torch.randint(0, 2, (n, d), generator=gen).float() * 2 - 1


def monomials(x, mons):
    return torch.stack([x[:, list(m)].prod(1) for m in mons], 1)


def run(cfg):
    target, opt_name, lr, d, seed = cfg
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    mons = TARGETS[target]

    net = torch.nn.Sequential(torch.nn.Linear(d, WIDTH), torch.nn.ReLU(), torch.nn.Linear(WIDTH, 1))
    opt = (torch.optim.SGD if opt_name == "sgd" else torch.optim.Adam)(net.parameters(), lr=lr)
    x_test = hypercube(8192, d, gen)
    m_test = monomials(x_test, mons)

    t_learn = [None] * len(mons)
    for step in range(MAX_STEPS + 1):
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                proj = (net(x_test) * m_test).mean(0)              # projection onto each monomial
            for i, p in enumerate(proj.tolist()):
                if t_learn[i] is None and p >= 0.5:
                    t_learn[i] = step
            if None not in t_learn:
                break
        x = hypercube(BATCH, d, gen)
        loss = ((net(x).squeeze(1) - monomials(x, mons).sum(1)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return dict(target=target, opt=opt_name, lr=lr, d=d, seed=seed, t_learn=t_learn)


if __name__ == "__main__":
    configs = [(t, o, lr, d, s) for t, o in product(TARGETS, LRS) for lr in LRS[o] for d in DIMS for s in SEEDS]
    results = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for r in pool.imap_unordered(run, configs):
            print(r, flush=True)
            results.append(r)
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=0)
