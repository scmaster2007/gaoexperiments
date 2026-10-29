"""Compare each trained network's implied prior with the exact prior, and test what explains the gap."""
import json, glob, os, argparse
import numpy as np
import pandas as pd
from scipy import stats
from common import load_prior, NIN

ap = argparse.ArgumentParser()
ap.add_argument("--prior", default="data/behaviours.csv")
ap.add_argument("--runs", default="runs/exp1_main")
ap.add_argument("--tag", default="")
args = ap.parse_args()
prior = load_prior(args.prior)
tt = prior["tt"].values.astype(np.int64)


# ---------- truth-table features (statistical simplicity, known to drive NN biases) ----------
def tt_bits(t):
    return np.array([(t >> x) & 1 for x in range(NIN)])


def walsh(f):
    s = 1 - 2 * f.astype(float)            # +-1
    h = s.copy(); n = 1
    while n < NIN:
        for i in range(0, NIN, 2 * n):
            a, b = h[i:i + n].copy(), h[i + n:i + 2 * n].copy()
            h[i:i + n], h[i + n:i + 2 * n] = a + b, a - b
        n *= 2
    return h / NIN                          # Fourier coefficients indexed by subset mask


def lz76(s):
    i, c, n = 0, 0, len(s)
    while i < n:
        l = 1
        while i + l <= n and s[i:i + l] in s[:i + l - 1]:
            l += 1
        c += 1; i += l
    return c


popcnt = np.array([bin(S).count("1") for S in range(NIN)])
feat = []
for t in tt:
    f = tt_bits(t)
    sens = np.array([[f[x] != f[x ^ (1 << k)] for k in range(4)] for x in range(NIN)])
    fh = walsh(f)
    nz = np.abs(fh) > 1e-9
    feat.append({
        "ones": f.sum(),
        "balance": abs(f.sum() - 8),
        "relevant_vars": int(sens.any(0).sum()),
        "avg_sensitivity": sens.sum(1).mean(),
        "fourier_degree": int(popcnt[nz].max()) if nz.any() else 0,
        "spectral_degree": float((popcnt * fh ** 2).sum()),     # energy-weighted degree (total influence)
        "lz": lz76("".join(map(str, f))),
    })
feat = pd.DataFrame(feat)
B = pd.concat([prior[["tt", "p", "log2p", "K", "short_min_meanT", "short_min_maxT", "mean_meanT", "Kt_mean", "witness"]], feat], axis=1)
B["log2_runtime"] = np.log2(B["short_min_meanT"] + 1)      # runtime of the fastest shortest program
B["log2_wruntime"] = np.log2(B["mean_meanT"] + 1)          # prior-weighted mean runtime over all programs

# ---------- load runs ----------
runs = {}
for js in sorted(glob.glob(f"{args.runs}/L*_s*.json")):
    name = os.path.basename(js)[:-5]
    meta = json.load(open(js))
    logq = np.load(f"{args.runs}/{name}_logq.npy")       # (N behaviours, R orders), log2 joint
    runs[name] = (meta, logq)
print("runs found:", list(runs))

rows, resid = [], {}
for name, (meta, logq) in runs.items():
    L, s = meta["args"]["layers"], meta["args"]["seed"]
    lq = logq.mean(1)
    r = lq - B["log2p"].values
    resid[name] = r
    leak = 1 - (2.0 ** logq).sum(0)               # joint mass on truth tables the prior says are impossible
    c = meta["curve"]
    rows.append({"run": name, "layers": L, "seed": s, "params": meta["params"],
                 "excess_bits_per_pred": c[-1]["excess_bits"],
                 "excess_at_half": [x["excess_bits"] for x in c if x["step"] == meta["args"]["steps"] // 2][0],
                 "leak_mass_mean": leak.mean(),
                 "order_sd_bits": logq.std(1).mean(),          # non-Bayesian: joint depends on input order
                 "train_min": meta["train_seconds"] / 60})
summary = pd.DataFrame(rows).sort_values(["layers", "seed"])
pd.set_option("display.width", 200)
print("\n=== Overall fit to the exact Bayes predictor ===")
print(summary.round(5).to_string(index=False))

R = pd.DataFrame(resid)
R["K"] = B["K"]
print("\n=== Implied minus true log2 prior, by shortest program length K (mean over behaviours) ===")
print(R.groupby("K").mean().round(2).to_string())
print("behaviours per K:", B.groupby("K").size().to_dict())

print("\n=== Seed consistency of the per-behaviour gap (Pearson r between seeds) ===")
for L in sorted(summary.layers.unique()):
    a, b = f"L{L}_s0", f"L{L}_s1"
    if a in R.columns and b in R.columns:
        allr = np.corrcoef(R[a], R[b])[0, 1]
        # also after removing the part explained by prior mass alone
        ra = R[a] - np.polyval(np.polyfit(B.log2p, R[a], 2), B.log2p)
        rb = R[b] - np.polyval(np.polyfit(B.log2p, R[b], 2), B.log2p)
        print(f"  layers={L}: r = {allr:.3f}   (after removing a quadratic in log2 prior: r = {np.corrcoef(ra, rb)[0,1]:.3f})")

# ---------- what explains the gap? ----------
feats_stat = ["avg_sensitivity", "relevant_vars", "fourier_degree", "spectral_degree", "balance", "lz"]
feats_alg = ["K", "log2_runtime", "log2_wruntime"]


def zs(x):
    return (x - x.mean()) / x.std()


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n, k = X1.shape
    s2 = ((y - yhat) ** 2).sum() / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X1.T @ X1)))
    return beta[1:], se[1:], r2


print("\n=== Regression of the gap on behaviour features (standardised coefficients; seed-averaged gap) ===")
reg_rows = []
for L in sorted(summary.layers.unique()):
    cols = [c for c in R.columns if c.startswith(f"L{L}_")]
    y = R[cols].mean(1).values
    base = np.column_stack([zs(B.log2p), zs(B.log2p) ** 2])
    stat = np.column_stack([zs(B[f]) for f in feats_stat])
    _, _, r2_p = ols(y, base)
    _, _, r2_ps = ols(y, np.column_stack([base, stat]))
    Xfull = np.column_stack([base, stat, zs(B.log2_runtime), zs(B.K)])
    beta, se, r2_full = ols(y, Xfull)
    names = ["log2p", "log2p^2"] + feats_stat + ["log2_runtime", "K"]
    co = dict(zip(names, beta)); ses = dict(zip(names, se))
    # partial correlation of runtime with the gap, controlling for prior mass, K and all statistical features
    ctrl = np.column_stack([base, stat, zs(B.K)])
    ry = y - np.column_stack([np.ones(len(y)), ctrl]) @ np.linalg.lstsq(np.column_stack([np.ones(len(y)), ctrl]), y, rcond=None)[0]
    xr = B.log2_runtime.values
    rx = xr - np.column_stack([np.ones(len(y)), ctrl]) @ np.linalg.lstsq(np.column_stack([np.ones(len(y)), ctrl]), xr, rcond=None)[0]
    pr, pp = stats.pearsonr(rx, ry)
    reg_rows.append({"layers": L, "R2_prior_only": r2_p, "R2_+stat_feats": r2_ps, "R2_+runtime,K": r2_full,
                     "beta_runtime": co["log2_runtime"], "se_runtime": ses["log2_runtime"],
                     "beta_sensitivity": co["avg_sensitivity"], "beta_relevant_vars": co["relevant_vars"],
                     "partial_r_runtime": pr, "partial_p": pp})
reg = pd.DataFrame(reg_rows)
print(reg.round(4).to_string(index=False))

# within-K rank correlations of runtime with gap (simple, assumption-light check)
print("\n=== Within each K: Spearman correlation of runtime with the gap (seed-averaged), K with >= 30 behaviours ===")
wk = []
for L in sorted(summary.layers.unique()):
    cols = [c for c in R.columns if c.startswith(f"L{L}_")]
    y = R[cols].mean(1)
    row = {"layers": L}
    for K, g in B.groupby("K"):
        if len(g) >= 30:
            row[f"K={K}"] = stats.spearmanr(g.log2_runtime, y[g.index])[0]
    wk.append(row)
print(pd.DataFrame(wk).round(3).to_string(index=False))

print("\n=== Placebo: shuffle runtime among behaviours with the same K, recompute partial r (200 shuffles) ===")
rng = np.random.default_rng(0)
ctrl = np.column_stack([np.ones(len(B)), zs(B.log2p), zs(B.log2p) ** 2] + [zs(B[f]) for f in feats_stat] + [zs(B.K)])
proj = lambda v: v - ctrl @ np.linalg.lstsq(ctrl, v, rcond=None)[0]
for L in sorted(summary.layers.unique()):
    cols = [c for c in R.columns if c.startswith(f"L{L}_")]
    ry = proj(R[cols].mean(1).values)
    real = stats.pearsonr(proj(B.log2_runtime.values), ry)[0]
    alt = stats.pearsonr(proj(B.log2_wruntime.values), ry)[0]
    null = []
    for _ in range(200):
        sh = B.log2_runtime.values.copy()
        for K, idx in B.groupby("K").indices.items():
            sh[idx] = rng.permutation(sh[idx])
        null.append(stats.pearsonr(proj(sh), ry)[0])
    null = np.array(null)
    print(f"  layers={L}: real partial r = {real:.3f}; with prior-weighted mean runtime instead = {alt:.3f}; "
          f"placebo mean {null.mean():.3f}, 99th pct of |placebo| = {np.quantile(np.abs(null), 0.99):.3f}")

B_out = B.copy()
for c in R.columns:
    if c != "K":
        B_out[f"gap_{c}"] = R[c]
B_out.to_csv(f"results/results_per_behaviour{args.tag}.csv", index=False)
summary.to_csv(f"results/results_summary{args.tag}.csv", index=False)
reg.to_csv(f"results/results_regression{args.tag}.csv", index=False)
print(f"\nwrote results/results_*{args.tag}.csv")
