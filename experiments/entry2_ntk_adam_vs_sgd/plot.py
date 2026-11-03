"""Figures from data/runs_compact.json, and data/summary_table.json (seed mean, min, max of each scalar).
usage: python plot.py data/runs_compact.json figures"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
runs = C["runs"]

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = [  # (label, opt, kernel, color) - fixed categorical order
    ("SGD: NTK  J·Jᵀ", "sgd", "K_std", "#2a78d6"),
    ("Adam: Adam kernel  J·diag(1/(√v̂+ε))·Jᵀ", "adam", "K_adam", "#eb6834"),
    ("Adam: Adam kernel, coordinates with √v̂<ε removed", "adam", "K_live", "#1baf7a"),
    ("Adam: NTK  J·Jᵀ", "adam", "K_std", "#eda100"),
]
DEPTHS = sorted({int(n.split("_L")[1].split("_")[0]) for n in runs})


def group(opt, depth):
    return [r for n, r in runs.items() if n.startswith(f"kern_{opt}_L{depth}_")]


def series(opt, depth, kern, metric):
    """epochs, mean, min, max over seeds (by epoch value)."""
    vals = {}
    for r in group(opt, depth):
        k = r["kernels"].get(kern)
        if not k:
            continue
        for e, v in zip(k["epoch"], k[metric]):
            if v is not None:
                vals.setdefault(round(e), []).append(v)
    ep = sorted(vals)
    a = [np.array(vals[e], float) for e in ep]
    return (np.array(ep), np.array([x.mean() for x in a]), np.array([x.min() for x in a]),
            np.array([x.max() for x in a]), [len(x) for x in a])


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


METRICS = [("spd", "Change per epoch\n‖K(t)−K(prev)‖/‖K(prev)‖ ÷ epochs", True),
           ("chg_ref", "Change from reference\n‖K(t)−K(ref)‖/‖K(ref)‖", True),
           ("shp_spd", "Shape change per epoch\n(each K scaled to ‖K‖=1)", True),
           ("align", "Label alignment\n(250 train points)", False),
           ("krr", "Kernel-regression\ntest accuracy (250→250)", False)]
fig, axes = plt.subplots(len(METRICS), len(DEPTHS), figsize=(4.2 * len(DEPTHS), 2.5 * len(METRICS)),
                         sharex=True, squeeze=False)
fig.patch.set_facecolor(SURF)
for i, (m, lab, logy) in enumerate(METRICS):
    for j, L in enumerate(DEPTHS):
        ax = axes[i][j]
        style(ax)
        for name, opt, kern, col in SERIES:
            ep, mu, lo, hi, n = series(opt, L, kern, m)
            if not len(ep):
                continue
            x = np.maximum(ep, 0.5)  # epoch 0 drawn at 0.5 on the log axis
            if logy:  # the reference checkpoint itself is 0 for chg_ref; leave it off the log axis
                keep = lo > 0
                x, mu, lo, hi = x[keep], mu[keep], lo[keep], hi[keep]
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(x, mu, color=col, linewidth=2, label=name)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if i == 0:
            ax.set_title(f"Depth {L}", color=INK, fontsize=11)
        if j == 0:
            ax.set_ylabel(lab, color=INK2, fontsize=8.5)
        if i == len(METRICS) - 1:
            ax.set_xlabel("Epoch (log; epoch 0 drawn at 0.5)", color=INK2, fontsize=8.5)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.005))
fig.suptitle("Kernel trajectories, Adam (lr 1e-3) vs SGD (lr 0.01); line = mean of 3 seeds, band = min–max",
             color=INK, fontsize=12, y=1.035)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "kernels.png"), dpi=150, bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# Hessian and accuracy
fig, axes = plt.subplots(2, len(DEPTHS), figsize=(4.2 * len(DEPTHS), 5.4), sharex=True, squeeze=False)
fig.patch.set_facecolor(SURF)
for j, L in enumerate(DEPTHS):
    for opt, col, nm in (("sgd", "#2a78d6", "SGD"), ("adam", "#eb6834", "Adam")):
        rs = group(opt, L)
        hv = {}
        for r in rs:
            for e, top, dom in r["hessian"]:
                hv.setdefault(round(e), []).append(top)
        ep = sorted(hv)
        if ep:
            a = [np.array(hv[e]) for e in ep]
            x = np.maximum(np.array(ep), 0.5)
            ax = axes[0][j]; style(ax)
            ax.fill_between(x, [v.min() for v in a], [v.max() for v in a], color=col, alpha=0.15, linewidth=0)
            ax.plot(x, [v.mean() for v in a], color=col, linewidth=2, label=nm)
        tv = {}
        for r in rs:
            for e, v in zip(r["curves"]["epoch"], r["curves"]["test_acc"]):
                tv.setdefault(round(e), []).append(v)
        ep = sorted(tv)
        if ep:
            a = [np.array(tv[e]) for e in ep]
            x = np.maximum(np.array(ep), 0.5)
            ax = axes[1][j]; style(ax)
            ax.fill_between(x, [v.min() for v in a], [v.max() for v in a], color=col, alpha=0.15, linewidth=0)
            ax.plot(x, [v.mean() for v in a], color=col, linewidth=2, label=nm)
    axes[0][j].set_xscale("log"); axes[0][j].set_yscale("log")
    axes[0][j].set_title(f"Depth {L}", color=INK, fontsize=11)
    axes[1][j].set_xlabel("Epoch (log; epoch 0 drawn at 0.5)", color=INK2, fontsize=8.5)
axes[0][0].set_ylabel("Top Hessian eigenvalue\nof the training loss", color=INK2, fontsize=8.5)
axes[1][0].set_ylabel("Test accuracy", color=INK2, fontsize=8.5)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hessian_accuracy.png"), dpi=150, bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# scalar table
rows = []
for L in DEPTHS:
    for name, opt, kern, col in SERIES:
        ks = [r["kernels"][kern] for r in group(opt, L) if kern in r["kernels"]]
        if not ks:
            continue
        row = {"depth": L, "series": name, "n": len(ks)}
        for m in ("settle90", "settle90_shape", "path_ratio", "path_ratio_shape", "align_tv_ratio", "krr_tv_ratio",
                  "align_reversals", "krr_reversals", "late_spd", "late_shp_spd"):
            v = [k.get(m) for k in ks if k.get(m) is not None]
            row[m] = (float(np.mean(v)), float(np.min(v)), float(np.max(v))) if v else None
        for m in ("chg_ref", "shp_ref", "align", "krr", "norm"):
            v = [k[m][-1] for k in ks if k[m] and k[m][-1] is not None]
            row["final_" + m] = (float(np.mean(v)), float(np.min(v)), float(np.max(v))) if v else None
        rows.append(row)
for L in DEPTHS:
    for opt in ("sgd", "adam"):
        rs = group(opt, L)
        hs = [r["hessian"][-1][1] for r in rs if r["hessian"]]
        ta = [r["curves"]["test_acc"][-1] for r in rs if r["curves"]["test_acc"]]
        tr = [r["curves"]["train_acc"][-1] for r in rs if r["curves"]["train_acc"]]
        rows.append({"depth": L, "series": f"{opt} run", "n": len(rs), "finished": [r["finished"] for r in rs],
                     "final_hessian": hs, "final_test_acc": ta, "final_train_acc": tr,
                     "frac_below_eps_final": [r["curves"]["frac_sqrtv_below_eps"][-1] for r in rs],
                     "minutes": [round(r["seconds"] / 60, 1) if r["seconds"] else None for r in rs]})
TABLE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "summary_table.json")
json.dump(rows, open(TABLE, "w"), indent=1)
print("wrote", OUT, "and", TABLE)
