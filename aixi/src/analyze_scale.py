"""Fit the exposure threshold e* as a function of model size and training data, from the scaling sweep."""
import glob, json, re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

A = pd.read_csv("runs/exp4_scale/assignments.csv")
rng = np.random.default_rng(0)
rows = []
for js in sorted(glob.glob("runs/exp4_scale/d*.json")):
    meta = json.load(open(js))
    d, params = meta["args"]["d"], meta["params"]
    for c in meta["log"]:
        Nseq = c["sequences"]
        lq = np.load(f"runs/exp4_scale/d{d}_n{Nseq}_logq.npy").mean(1)
        A["log2q"] = lq[A.idx.values]
        q0_all = A[A.level == 0].log2q.values
        q0 = q0_all.mean()
        for lvl, g in A[A.level > 0].groupby("level"):
            freq = g.freq.iloc[0]
            target = np.log2(freq)
            r = (g.log2q.mean() - q0) / (target - q0)
            # bootstrap over behaviours (both the planted group and the never-shown group)
            bs = []
            for _ in range(500):
                gm = rng.choice(g.log2q.values, len(g)).mean()
                q0b = rng.choice(q0_all, len(q0_all)).mean()
                bs.append((gm - q0b) / (target - q0b))
            rows.append({"d": d, "params": params, "sequences": Nseq, "excess": c["excess_bits"], "level": lvl,
                         "freq": freq, "exposure": freq * Nseq, "default_log2q": q0, "target_log2p": target,
                         "mean_log2q": g.log2q.mean(), "response": r, "resp_lo": np.quantile(bs, 0.05),
                         "resp_hi": np.quantile(bs, 0.95)})
R = pd.DataFrame(rows)
R.to_csv("results/results_scale_points.csv", index=False)
pd.set_option("display.width", 220)
print("excess loss vs exact Bayes (bits/pred) by width and training sequences:")
print(R.groupby(["params", "sequences"]).excess.first().unstack().round(5).to_string())
print("\nresponse (0 = belief stays at never-seen default, 1 = matches true frequency):")
print(R.pivot_table(index=["params", "sequences"], columns="exposure", values="response").round(2).to_string()
      if R.exposure.nunique() < 12 else
      R.pivot_table(index=["params", "level"], columns="sequences", values="response").round(2).to_string())


def sig(loge, logestar, k):
    return 1.0 / (1.0 + 2.0 ** (-k * (loge - logestar)))


# Fit in bits: mean log2 q(level) = q0 + (log2 freq - q0) * sig(log2 exposure - log2 e*), q0 = never-shown mean.
def pred(df, ls, k):
    return df.default_log2q + (df.target_log2p - df.default_log2q) * sig(np.log2(df.exposure), ls, k)


fits = []
for (params, Nseq), g in R.groupby(["params", "sequences"]):
    sol = least_squares(lambda th: pred(g, th[0], th[1]) - g.mean_log2q, x0=[10, 1], bounds=([0, 0.05], [20, 10]))
    rss = (sol.fun ** 2).sum()
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * rss / max(len(g) - 2, 1)
    fits.append({"params": params, "sequences": Nseq, "default_log2q": g.default_log2q.iloc[0], "log2_estar": sol.x[0],
                 "se": np.sqrt(cov[0, 0]), "estar": 2 ** sol.x[0], "k": sol.x[1], "rmse_bits": np.sqrt(rss / len(g))})
Fz = pd.DataFrame(fits)
print("\nthreshold e* per width and data scale (fit in bits; e* bounded to [1, 2^20]):")
print(Fz.round(3).to_string(index=False))
Fz.to_csv("results/results_scale_thresholds.csv", index=False)

x_p, x_n = np.log2(R.params.values), np.log2(R.sequences.values)
y = R.mean_log2q.values
tot = ((y - y.mean()) ** 2).sum()
out = {}
for name, up, un in [("one threshold for all", False, False), ("+ model size", True, False),
                     ("+ data scale", False, True), ("+ model size + data scale", True, True)]:
    def res(th):
        c, al, be, k = th
        ls = c + (al * (x_p - x_p.mean()) if up else 0) + (be * (x_n - x_n.mean()) if un else 0)
        return pred(R, ls, k).values - y
    sol = least_squares(res, x0=[10, 0, 0, 1])
    rss = (sol.fun ** 2).sum()
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * rss / (len(y) - 4)
    # R2 relative to a model in which beliefs never move from the default (sig = 0)
    rss0 = ((R.default_log2q.values - y) ** 2).sum()
    out[name] = (sol.x, np.sqrt(np.diag(cov)), 1 - rss / rss0, np.sqrt(rss / len(y)))
print("\nglobal fits in bits (R2 is relative to 'belief never moves from the default'):")
for name, (th, se, r2, rmse) in out.items():
    print(f"  {name:28s} R2 {r2:.3f}  RMSE {rmse:.2f} bits | log2 e* {th[0]:.2f}  alpha {th[1]:+.3f}±{se[1]:.3f}  "
          f"beta {th[2]:+.3f}±{se[2]:.3f}  k {th[3]:.2f}")
th = out["+ model size + data scale"][0]
print(f"\nimplied law: e* ≈ {2 ** (th[0] - th[1] * x_p.mean() - th[2] * x_n.mean()):.3g} · params^{th[1]:.2f} · sequences^{th[2]:.2f}")
json.dump({k: {"theta": v[0].tolist(), "se": v[1].tolist(), "R2": v[2], "RMSE": v[3]} for k, v in out.items()},
          open("results/results_scale_law.json", "w"), indent=1)
