"""Run on the GPU node: summarise Pythia runs into small CSVs (points, thresholds, default-belief correlations)."""
import glob, json, sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import least_squares

D = np.load("runs/exp4_scale/law_data.npz")
beh = pd.read_csv("data/behaviours.csv")
NIN = 16
tt = D["tt"]
nat_log2p = np.log2(D["natural_p"])
plant = pd.DataFrame({"idx": D["plant_idx"], "level": D["plant_level"], "freq": D["plant_freq"]})
planted = set(plant.idx)

# behaviour features (same definitions as the CPU analysis)
popcnt = np.array([bin(S).count("1") for S in range(NIN)])
feat = []
for t in tt:
    f = np.array([(t >> x) & 1 for x in range(NIN)])
    sens = np.array([[f[x] != f[x ^ (1 << k)] for k in range(4)] for x in range(NIN)])
    s = 1 - 2 * f.astype(float); h = s.copy(); n = 1
    while n < NIN:
        for i in range(0, NIN, 2 * n):
            u, v = h[i:i + n].copy(), h[i + n:i + 2 * n].copy(); h[i:i + n], h[i + n:i + 2 * n] = u + v, u - v
        n *= 2
    fh = h / NIN
    feat.append({"avg_sensitivity": sens.sum(1).mean(), "relevant_vars": int(sens.any(0).sum()),
                 "spectral_degree": float((popcnt * fh ** 2).sum()), "ones": f.sum()})
F = pd.DataFrame(feat)
F["log2_runtime"] = np.log2(beh.short_min_meanT.values + 1)
F["K"] = beh.K.values
F["natural_log2p"] = nat_log2p

rows, defaults, runs = [], [], []
for js in sorted(glob.glob("runs/lm/*.json")):
    meta = json.load(open(js)); name = js[len("runs/lm/"):-5]
    params, scratch = meta["params"], meta["args"]["scratch"]
    for c in meta["log"]:
        Nseq = c["sequences"]
        lq = np.load(f"runs/lm/{name}_n{Nseq}_logq.npy").mean(1)
        runs.append({"run": name, "params": params, "scratch": scratch, "sequences": Nseq, "excess": c["excess_bits"],
                     "order_sd": float(np.load(f"runs/lm/{name}_n{Nseq}_logq.npy").std(1).mean())})
        # never-seen natural behaviours: not planted and expected exposure < 0.05 at this data scale
        p_tr = D["p"] / D["p"].sum()
        unseen = np.array([(i not in planted) and (p_tr[i] * max(Nseq, 1) < 0.05) for i in range(len(tt))])
        drow = {"run": name, "params": params, "scratch": scratch, "sequences": Nseq, "n_unseen": int(unseen.sum()),
                "mean_log2q_unseen": float(lq[unseen].mean())}
        for f in ["avg_sensitivity", "log2_runtime", "natural_log2p", "K", "relevant_vars", "spectral_degree"]:
            drow[f"rho_{f}"] = stats.spearmanr(F[f].values[unseen], lq[unseen])[0]
        # partial: runtime and algorithmic probability after sensitivity, relevant vars, spectral degree, K
        X = np.column_stack([np.ones(unseen.sum())] + [F[f].values[unseen] for f in ["avg_sensitivity", "relevant_vars", "spectral_degree", "K"]])
        pr = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        drow["partial_r_runtime"] = stats.pearsonr(pr(F.log2_runtime.values[unseen]), pr(lq[unseen]))[0]
        drow["partial_r_natural_log2p"] = stats.pearsonr(pr(F.natural_log2p.values[unseen]), pr(lq[unseen]))[0]
        defaults.append(drow)
        if Nseq == 0:
            continue
        plant["log2q"] = lq[plant.idx.values]
        q0 = plant[plant.level == 0].log2q.values
        for lvl, g in plant[plant.level > 0].groupby("level"):
            fr = g.freq.iloc[0]; tgt = np.log2(fr)
            rows.append({"run": name, "params": params, "scratch": scratch, "sequences": Nseq, "level": lvl, "freq": fr,
                         "exposure": fr * Nseq, "default_log2q": q0.mean(), "target_log2p": tgt,
                         "mean_log2q": g.log2q.mean(), "sd_log2q": g.log2q.std(),
                         "response": (g.log2q.mean() - q0.mean()) / (tgt - q0.mean())})
R = pd.DataFrame(rows); R.to_csv("results/lm/lm_points.csv", index=False)
pd.DataFrame(runs).to_csv("results/lm/lm_runs.csv", index=False)
pd.DataFrame(defaults).to_csv("results/lm/lm_defaults.csv", index=False)


def sig(le, ls, k):
    return 1 / (1 + 2.0 ** (-k * (le - ls)))


def pred(df, ls, k):
    return df.default_log2q + (df.target_log2p - df.default_log2q) * sig(np.log2(df.exposure), ls, k)


fits = []
for (run, Nseq), g in R.groupby(["run", "sequences"]):
    sol = least_squares(lambda th: pred(g, th[0], th[1]) - g.mean_log2q, x0=[10, 1], bounds=([0, 0.05], [20, 10]))
    rss = (sol.fun ** 2).sum(); cov = np.linalg.pinv(sol.jac.T @ sol.jac) * rss / max(len(g) - 2, 1)
    fits.append({"run": run, "params": g.params.iloc[0], "scratch": g.scratch.iloc[0], "sequences": Nseq,
                 "log2_estar": sol.x[0], "se": float(np.sqrt(cov[0, 0])), "k": sol.x[1], "rmse_bits": float(np.sqrt(rss / len(g)))})
T = pd.DataFrame(fits); T.to_csv("results/lm/lm_thresholds.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(runs).round(4).to_string(index=False))
print(T.round(3).to_string(index=False))
print(pd.DataFrame(defaults).round(3).to_string(index=False))
