"""Summarise lm_loops.py outputs: for each model and step count, fit to exact Bayes and what the in-context prior tracks."""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from common import load_prior, NIN

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/recur"
prior = load_prior("data/behaviours.csv")
tt = prior.tt.values.astype(np.int64)
lp = prior.log2p.values
K = prior.K.values
runtime = np.log2(prior.short_min_meanT.values + 1)
popcnt = np.array([bin(S).count("1") for S in range(NIN)])
feat = []
for t in tt:
    f = np.array([(t >> x) & 1 for x in range(NIN)])
    sens = np.array([[f[x] != f[x ^ (1 << k)] for k in range(4)] for x in range(NIN)])
    h = 1 - 2 * f.astype(float); n = 1
    while n < NIN:
        for i in range(0, NIN, 2 * n):
            u, v = h[i:i + n].copy(), h[i + n:i + 2 * n].copy(); h[i:i + n], h[i + n:i + 2 * n] = u + v, u - v
        n *= 2
    fh = h / NIN
    feat.append({"S": sens.sum(1).mean(), "relevant_vars": int(sens.any(0).sum()),
                 "spectral_degree": float((popcnt * fh ** 2).sum()), "balance": abs(f.sum() - 8)})
F = pd.DataFrame(feat)
S = F.S.values
stat = [F[c].values for c in F.columns]


def pcor(y, x, ctrl):
    y, x = stats.rankdata(y), stats.rankdata(x)
    C = np.column_stack([np.ones(len(y))] + [stats.rankdata(c) for c in ctrl])
    r = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return stats.pearsonr(r(y), r(x))[0]


rows = []
for js in sorted(glob.glob(f"{DIR}/*.json")):
    meta = json.load(open(js)); name = os.path.basename(js)[:-5]
    z = np.load(f"{DIR}/{name}.npz")
    for k, r in enumerate(z["steps"]):
        q = z["logq"][k].mean(1); ex = z["excess_pos"][k]
        rows.append({"model": name, "steps": int(r), "params_M": meta["params"] / 1e6,
                     "excess": ex.mean(), "excess_pos1-4": ex[:4].mean(), "excess_pos13-16": ex[12:].mean(),
                     "seq_excess_bits": ex.sum(), "mean_gap": (q - lp).mean(),
                     "rho_S": stats.spearmanr(q, S)[0], "rho_logP": stats.spearmanr(q, lp)[0],
                     "p_S|logP": pcor(q, S, [lp]), "p_logP|S": pcor(q, lp, [S]),
                     "p_logP|stat": pcor(q, lp, stat), "p_K|stat": pcor(q, K, stat),
                     "p_runtime": pcor(q, runtime, stat + [lp, K]),
                     "order_sd": z["logq"][k].std(1).mean()})
D = pd.DataFrame(rows)
D.to_csv(f"results/results_{os.path.basename(DIR)}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(D.round(4).to_string(index=False))
