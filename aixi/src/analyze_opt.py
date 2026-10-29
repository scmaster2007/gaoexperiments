"""Compare training procedures by how close the learned predictor is to the exact Bayes (bounded Solomonoff) predictor."""
import json, glob, os
import numpy as np
import pandas as pd
from scipy import stats

F = pd.read_csv("results/results_per_behaviour.csv")          # behaviour features for machine A (from analyze.py)
F = F[[c for c in F.columns if not c.startswith("gap_")]]
feats_stat = ["avg_sensitivity", "relevant_vars", "fourier_degree", "spectral_degree", "balance", "lz"]
z = lambda v: (v - v.mean()) / v.std()
Xctrl = np.column_stack([np.ones(len(F)), z(F.log2p), z(F.log2p) ** 2] + [z(F[f]) for f in feats_stat] + [z(F.K)])
proj = lambda v: v - Xctrl @ np.linalg.lstsq(Xctrl, v, rcond=None)[0]
Xfull = np.column_stack([Xctrl, z(F.log2_runtime)])
bins = {"K<=5": F.K <= 5, "K6-9": (F.K >= 6) & (F.K <= 9), "K10-11": (F.K >= 10) & (F.K <= 11), "K12-13": F.K >= 12}

# (path prefix, condition label, compute in units of the 5k-step baseline step count)
runs = []
for p in sorted(glob.glob("runs/exp2_optimizer/*_s[0-9].json")):
    runs.append((p[:-5], os.path.basename(p).rsplit("_s", 1)[0], None))
for p in sorted(glob.glob("runs/exp1_main/L2_s[0-9].json")):
    runs.append((p[:-5], "base_10k_steps", 2.0))
for p in sorted(glob.glob("runs/exp1_long/L2_s[0-9].json")):
    runs.append((p[:-5], "base_30k_steps", 6.0))

base_time = {}
for prefix, cond, _ in runs:
    if cond == "base":
        base_time[json.load(open(prefix + ".json"))["args"]["seed"]] = json.load(open(prefix + ".json"))["train_seconds"]

rows = []
for prefix, cond, comp in runs:
    meta = json.load(open(prefix + ".json"))
    logq = np.load(prefix + "_logq.npy")
    gap = logq.mean(1) - F.log2p.values
    seed = meta["args"]["seed"]
    if comp is None:
        comp = meta["train_seconds"] / base_time.get(seed, meta["train_seconds"])
    beta = np.linalg.lstsq(Xfull, gap, rcond=None)[0]
    r = {"condition": cond, "seed": seed, "compute_x": comp,
         "excess_bits_per_pred": meta["curve"][-1]["excess_bits"],
         "mean_abs_gap_all": np.abs(gap).mean(),
         "order_sd_bits": logq.std(1).mean(),
         "runtime_partial_r": stats.pearsonr(proj(F.log2_runtime.values), proj(gap))[0],
         "runtime_beta": beta[-1], "sensitivity_beta": beta[3]}
    for b, m in bins.items():
        r[f"gap_{b}"] = gap[m.values].mean()
        r[f"absgap_{b}"] = np.abs(gap[m.values]).mean()
    rows.append(r)
    np.save(prefix + "_gap.npy", gap)

df = pd.DataFrame(rows)
order = ["base", "is", "soft", "exch", "sam", "base_10k_steps", "base_30k_steps"]
df["condition"] = pd.Categorical(df.condition, order)
df.to_csv("results/results_optimizer_per_run.csv", index=False)
agg = df.groupby("condition", observed=True).agg(["mean", "min", "max"])
pd.set_option("display.width", 250)
cols = ["compute_x", "excess_bits_per_pred", "mean_abs_gap_all", "gap_K<=5", "gap_K6-9", "gap_K10-11", "gap_K12-13",
        "absgap_K6-9", "runtime_partial_r", "runtime_beta", "sensitivity_beta", "order_sd_bits"]
print("Per condition: mean over seeds  [min, max]")
for c in cols:
    print(f"\n{c}")
    print(agg[c].round(4).to_string())
agg.to_csv("results/results_optimizer_summary.csv")

# paired per-behaviour comparison against the 5k baseline (seed-averaged)
print("\n=== Per-behaviour |gap| change vs 5k-step baseline (seed-averaged gaps), by K bin ===")
G = {c: np.mean([np.load(f"runs/exp2_optimizer/{c}_s{s}_gap.npy") for s in (0, 1) if os.path.exists(f"runs/exp2_optimizer/{c}_s{s}_gap.npy")], 0)
     for c in ["base", "is", "soft", "exch", "sam"] if glob.glob(f"runs/exp2_optimizer/{c}_s*_gap.npy")}
out = []
for c, g in G.items():
    row = {"condition": c}
    for b, m in bins.items():
        row[b] = (np.abs(g[m.values]) - np.abs(G["base"][m.values])).mean()
    out.append(row)
print(pd.DataFrame(out).round(3).to_string(index=False))
