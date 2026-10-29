"""Looped vs untied learners: fit to exact Bayes, and what sets the belief about never-seen behaviours."""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from common import load_prior, NIN

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs/loop"
prior = load_prior("data/behaviours.csv")
tt = prior.tt.values.astype(np.int64)
lp = prior.log2p.values
K = prior.K.values
popcnt = np.array([bin(S).count("1") for S in range(NIN)])


def walsh(f):
    h = 1 - 2 * f.astype(float); n = 1
    while n < NIN:
        for i in range(0, NIN, 2 * n):
            u, v = h[i:i + n].copy(), h[i + n:i + 2 * n].copy(); h[i:i + n], h[i + n:i + 2 * n] = u + v, u - v
        n *= 2
    return h / NIN


def lz76(s):
    i, c, n = 0, 0, len(s)
    while i < n:
        l = 1
        while i + l <= n and s[i:i + l] in s[:i + l - 1]:
            l += 1
        c += 1; i += l
    return c


rows = []
for t in tt:
    f = np.array([(t >> x) & 1 for x in range(NIN)])
    sens = np.array([[f[x] != f[x ^ (1 << k)] for k in range(4)] for x in range(NIN)])
    fh = walsh(f); nz = np.abs(fh) > 1e-9
    rows.append({"avg_sensitivity": sens.sum(1).mean(), "relevant_vars": int(sens.any(0).sum()),
                 "fourier_degree": int(popcnt[nz].max()) if nz.any() else 0,
                 "spectral_degree": float((popcnt * fh ** 2).sum()), "balance": abs(f.sum() - 8),
                 "lz": lz76("".join(map(str, f)))})
F = pd.DataFrame(rows)
runtime = np.log2(prior.short_min_meanT.values + 1)
S = F.avg_sensitivity.values
stat_names = ["avg_sensitivity", "relevant_vars", "fourier_degree", "spectral_degree", "balance", "lz"]
zs = lambda x: (x - x.mean()) / x.std()
unseen = prior.p.values * 5.12e6 < 0.05          # expected exposure < 0.05 even in the longest run


def pcor(y, x, ctrl):
    """Rank partial correlation of y and x given the columns of ctrl."""
    y, x = stats.rankdata(y), stats.rankdata(x)
    C = np.column_stack([np.ones(len(y))] + [stats.rankdata(c) for c in ctrl])
    r = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return stats.pearsonr(r(y), r(x))[0]


base = np.column_stack([zs(lp), zs(lp) ** 2])
stat = np.column_stack([zs(F[f].values) for f in stat_names])
X = np.column_stack([np.ones(len(lp)), base, stat, zs(runtime), zs(K)])
names = ["c", "log2p", "log2p2"] + stat_names + ["log2_runtime", "K"]
Xr = np.column_stack([np.ones(len(lp)), base, stat])          # for the runtime profile of the residual gap
rt_raw = prior.short_min_meanT.values
edges = [0, 8, 10, 12, 14, 25]                                 # mean steps of the fastest shortest program (quintile-like)
out, prof = [], []
for js in sorted(glob.glob(f"{RUNS}/*.json")):
    meta = json.load(open(js)); name = os.path.basename(js)[:-5]
    cfg, seed = name.rsplit("_s", 1)
    for L, ex in meta["excess"].items():
        logq = np.load(f"{RUNS}/{name}_r{L}_logq.npy"); q = logq.mean(1); gap = q - lp
        beta = np.linalg.lstsq(X, gap, rcond=None)[0]; co = dict(zip(names, beta))
        u = unseen
        out.append({"config": cfg, "seed": int(seed), "eval_loops": int(L), "params": meta["params"],
                    "compute": meta["compute"] / 1.28e6 / 2, "excess": ex,
                    "absgap_K6_9": np.abs(gap[(K >= 6) & (K <= 9)]).mean(), "gap_K12_13": gap[K >= 12].mean(),
                    "order_sd": logq.std(1).mean(),
                    "beta_sens": co["avg_sensitivity"], "beta_runtime": co["log2_runtime"],
                    "U_rho_sens": stats.spearmanr(q[u], S[u])[0], "U_rho_logP": stats.spearmanr(q[u], lp[u])[0],
                    "U_p_sens|logP": pcor(q[u], S[u], [lp[u]]), "U_p_logP|sens": pcor(q[u], lp[u], [S[u]]),
                    "U_p_logP|stat": pcor(q[u], lp[u], [F[f].values[u] for f in stat_names]),
                    "U_p_runtime": pcor(q[u], runtime[u], [F[f].values[u] for f in stat_names] + [lp[u], K[u]])})
        res = gap - Xr @ np.linalg.lstsq(Xr, gap, rcond=None)[0]
        b = np.digitize(rt_raw, edges[1:-1])
        prof.append({"config": cfg, "seed": int(seed), "eval_loops": int(L),
                     **{f"steps_{edges[i]}-{edges[i+1]}": res[b == i].mean() for i in range(len(edges) - 1)}})
D = pd.DataFrame(out)
P = pd.DataFrame(prof)
D.to_csv(f"results/results_{os.path.basename(RUNS)}_runs.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(f"never-seen behaviours: {unseen.sum()}")
print(D.groupby(["config", "eval_loops"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
print("\nresidual gap (bits) by runtime of the fastest shortest program, after prior mass and statistical features:")
print(P.groupby(["config", "eval_loops"]).mean(numeric_only=True).drop(columns="seed").round(2).to_string())
