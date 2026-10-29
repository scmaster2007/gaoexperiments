"""Planted-exposure experiment: does the learned prior follow a law in (exposure, zero-exposure belief)?"""
import json
import numpy as np
import pandas as pd
from scipy import stats

NTRAIN = 5000 * 128
F = pd.read_csv("results/results_per_behaviour.csv")
A = pd.read_csv("runs/exp3_plant/assignments.csv")
LEVELS = sorted(A.exposure.unique())
logq = {m: np.load(f"runs/exp3_plant/m{m}_logq.npy").mean(1) for m in range(6)}
pv = {m: np.load(f"runs/exp3_plant/pvec_m{m}.npy") for m in range(6)}
excess = {m: json.load(open(f"runs/exp3_plant/m{m}.json"))["curve"][-1]["excess_bits"] for m in range(6)}
print("overall excess (bits/pred) per model:", {m: round(v, 4) for m, v in excess.items()})

A["log2q"] = [logq[m][i] for m, i in zip(A.model, A.idx)]
A["log2p_train"] = np.where(A.exposure > 0, np.log2(np.maximum(A.exposure, 1e-300) / NTRAIN), -np.inf)
A["gap"] = A.log2q - A.log2p_train
zero = A[A.exposure == 0].set_index("idx")["log2q"]          # belief about the behaviour when never shown
A["log2q_zero"] = A.idx.map(zero)

print("\n=== Dose-response for the 72 planted behaviours (each appears once at every level) ===")
tab = A.groupby("exposure").agg(train_log2p=("log2p_train", "mean"), mean_log2q=("log2q", "mean"),
                                sd_log2q=("log2q", "std"), mean_gap=("gap", "mean"), sd_gap=("gap", "std")).round(2)
print(tab.to_string())
print("natural log2 P of these behaviours (mean):", round(A.natural_log2p.mean(), 2))

# natural (non-planted) behaviours in the same models, binned by exposure, for the collapse test
planted = set(A.idx)
nat = []
for m in range(6):
    p = pv[m]
    for i in range(len(p)):
        if i in planted or p[i] <= 0:
            continue
        nat.append((m, i, NTRAIN * p[i], logq[m][i] - np.log2(p[i]), int(F.K[i])))
nat = pd.DataFrame(nat, columns=["model", "idx", "expo", "gap", "K"])
print("\n=== Collapse test: planted (K 10-11) vs natural behaviours at matched exposure (within 2x) ===")
rows = []
for e in LEVELS[1:]:
    sel = nat[(nat.expo >= e / 2) & (nat.expo <= e * 2)]
    pl = A[A.exposure == e]
    t = stats.ttest_ind(pl.gap, sel.gap, equal_var=False) if len(sel) > 2 else None
    rows.append({"exposure": e, "planted_gap": pl.gap.mean(), "planted_K": pl.K.mean(),
                 "natural_gap": sel.gap.mean(), "natural_K": sel.K.mean(), "n_natural": len(sel),
                 "diff": pl.gap.mean() - sel.gap.mean(), "p": t.pvalue if t else np.nan})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== Does the zero-exposure belief predict the learned belief at higher exposure? ===")
print("log2 q(b, e) = a_e + beta_e * log2 q(b, 0) across the 72 behaviours (q(b,0) measured in a different model)")
rows = []
for e in LEVELS[1:]:
    s = A[A.exposure == e]
    sl, ic, r, p, se = stats.linregress(s.log2q_zero, s.log2q)
    rows.append({"exposure": e, "beta": sl, "se": se, "r": r, "p": p})
fit = pd.DataFrame(rows)
print(fit.round(3).to_string(index=False))

# noise ceiling: how well does one model's belief about a natural rare behaviour predict another model's?
rare = nat[nat.expo < 0.05]
piv = rare.pivot_table(index="idx", columns="model", values="gap")
cors = [piv[a].corr(piv[b]) for a in range(6) for b in range(a + 1, 6)]
print(f"\nnoise ceiling: between-model correlation of beliefs about never-seen natural behaviours: mean r = {np.mean(cors):.3f}")

print("\n=== What sets the zero-exposure belief? (72 behaviours) ===")
z0 = A[A.exposure == 0]
for f in ["log2_runtime", "avg_sensitivity", "natural_log2p", "K"]:
    r, p = stats.spearmanr(z0[f], z0.log2q)
    print(f"  Spearman(log2 q at zero exposure, {f}) = {r:.3f} (p={p:.3g})")
print("\n=== Runtime effect by exposure level (Spearman of runtime with log2 q, 72 behaviours) ===")
for e in LEVELS:
    s = A[A.exposure == e]
    r, p = stats.spearmanr(s.log2_runtime, s.log2q)
    print(f"  exposure {e:5d}: r = {r:.3f} (p={p:.3g})")

# simple predictive law, evaluated with leave-one-behaviour-out:
#   log2 q(b,e) = w(e) * log2 Ptrain(b,e) + (1 - w(e)) * log2 q(b,0) + c(e)
print("\n=== Interpolation law, leave-one-behaviour-out prediction of log2 q(b, e) for e > 0 ===")
pos = A[A.exposure > 0].copy()
preds_law, preds_expo, preds_P, y = [], [], [], []
for b in pos.idx.unique():
    tr, te = pos[pos.idx != b], pos[pos.idx == b]
    for _, row in te.iterrows():
        t = tr[tr.exposure == row.exposure]
        X = np.column_stack([np.ones(len(t)), t.log2q_zero])
        coef = np.linalg.lstsq(X, t.log2q, rcond=None)[0]
        preds_law.append(coef[0] + coef[1] * row.log2q_zero)
        preds_expo.append(t.log2q.mean())                  # exposure-only model
        preds_P.append(row.log2p_train)                     # "network = exact Bayes" model
        y.append(row.log2q)
y = np.array(y)
for name, pr in [("exact prior (network = Bayes)", preds_P), ("exposure only", preds_expo), ("exposure + zero-exposure belief", preds_law)]:
    pr = np.array(pr)
    print(f"  {name:34s}: RMSE {np.sqrt(np.mean((y - pr) ** 2)):.2f} bits,  R2 {1 - np.sum((y - pr) ** 2) / np.sum((y - y.mean()) ** 2):.3f}")
A.to_csv("results/results_plant.csv", index=False)
fit.to_csv("results/results_plant_fit.csv", index=False)
