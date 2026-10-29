"""Figure for the exposure-threshold scaling sweep."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

R = pd.read_csv("results/results_scale_points.csv")
T = pd.read_csv("results/results_scale_thresholds.csv")
T = T[T.se < 2]
SIZES = sorted(R.params.unique())
COL = dict(zip(SIZES, ["#86b6ef", "#2a78d6", "#104281"]))       # ordinal blue ramp, validated
LBL = {p: f"{p/1000:.0f}k params" for p in SIZES}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)


def sig(le, ls, k):
    return 1 / (1 + 2.0 ** (-k * (le - ls)))


# (a) response vs exposure; levels whose true probability is within 3 bits of the default are left out (ratio unstable)
a = ax[0]
for p in SIZES:
    g = R[(R.params == p) & ((R.target_log2p - R.default_log2q) >= 3)]
    a.errorbar(g.exposure, g.response, yerr=[g.response - g.resp_lo, g.resp_hi - g.response], fmt="o", ms=5,
               color=COL[p], ecolor=COL[p], elinewidth=1, alpha=0.9, mec=SURF, mew=1, label=LBL[p], zorder=3)
    gall = R[R.params == p]
    def res(th):
        return (gall.default_log2q + (gall.target_log2p - gall.default_log2q) * sig(np.log2(gall.exposure), th[0], th[1])
                - gall.mean_log2q).values
    th = least_squares(res, [10, 1]).x
    xs = np.logspace(0, 12.5, 200, base=2)
    a.plot(xs, sig(np.log2(xs), *th), color=COL[p], lw=2, zorder=2)
a.set_xscale("log", base=2); a.set_xlim(2 ** 1.5, 2 ** 12.5); a.set_ylim(-0.25, 1.05)
a.axhline(0, color=GRID, lw=1, zorder=0); a.axhline(1, color=GRID, lw=1, zorder=0); a.axhline(0.5, color=GRID, lw=1, ls=":", zorder=0)
a.set_xticks([4, 16, 64, 256, 1024, 4096]); a.set_xticklabels(["4", "16", "64", "256", "1,024", "4,096"])
a.set_xlabel("times the behaviour appeared in training (exposure)")
a.set_ylabel("fraction of the way from never-seen default\n(0) to true training probability (1)")
a.set_title("(a) Beliefs follow exposure: one S-curve per model size", loc="left")
a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98), labelcolor=INK)

# (b) threshold vs model size
b = ax[1]
for p in SIZES:
    g = T[T.params == p]
    jitter = 2 ** (np.linspace(-0.08, 0.08, len(g)))
    b.errorbar(g.params * jitter, g.estar, yerr=[g.estar - 2 ** (g.log2_estar - g.se), 2 ** (g.log2_estar + g.se) - g.estar],
               fmt="o", ms=6, color=COL[p], ecolor=COL[p], elinewidth=1, mec=SURF, mew=1, zorder=3)
X = np.column_stack([np.ones(len(T)), np.log2(T.params), np.log2(T.sequences)])
w = np.sqrt(1 / T.se.values ** 2)
beta = np.linalg.lstsq(X * w[:, None], T.log2_estar.values * w, rcond=None)[0]
xs = np.logspace(np.log2(20000), np.log2(500000), 50, base=2)
mid = np.log2(T.sequences).mean()
b.plot(xs, 2 ** (beta[0] + beta[1] * np.log2(xs) + beta[2] * mid), color=INK2, lw=2, zorder=2)
b.set_xscale("log", base=2); b.set_yscale("log", base=2)
b.set_xticks(SIZES); b.set_xticklabels([LBL[p].replace(" params", "") for p in SIZES])
b.set_yticks([256, 512, 1024, 2048]); b.set_yticklabels(["256", "512", "1,024", "2,048"])
b.set_ylim(180, 2600)
b.set_xlabel("model parameters"); b.set_ylabel("threshold e* (exposures to move halfway)")
b.set_title("(b) Larger models need fewer exposures", loc="left")
b.text(xs[28], 2 ** (beta[0] + beta[1] * np.log2(xs[28]) + beta[2] * mid) * 1.25,
       f"e* ∝ params^{beta[1]:.2f}", color=INK2, fontsize=9.5)
b.text(21000, 200, "each point: one data scale (128k–2M sequences); bars: ±1 s.e.", color=INK2, fontsize=8)
b.grid(True, which="major", color=GRID, lw=0.8); b.set_axisbelow(True)

# (c) threshold vs training data, per size
c = ax[2]
for p in SIZES:
    g = T[T.params == p].sort_values("sequences")
    c.errorbar(g.sequences, g.estar, yerr=[g.estar - 2 ** (g.log2_estar - g.se), 2 ** (g.log2_estar + g.se) - g.estar],
               fmt="-o", ms=6, lw=2, color=COL[p], ecolor=COL[p], elinewidth=1, mec=SURF, mew=1)
    c.text(g.sequences.iloc[-1] * 1.12, g.estar.iloc[-1], LBL[p], color=INK, fontsize=9, va="center")
c.set_xscale("log", base=2); c.set_yscale("log", base=2)
c.set_xticks([128000, 256000, 512000, 1024000, 2048000]); c.set_xticklabels(["128k", "256k", "512k", "1M", "2M"])
c.set_yticks([256, 512, 1024, 2048]); c.set_yticklabels(["256", "512", "1,024", "2,048"])
c.set_xlim(100000, 5200000); c.set_ylim(180, 2600)
c.set_xlabel("training sequences seen"); c.set_ylabel("threshold e*")
c.set_title("(c) More data lowers e* only for larger models", loc="left")
c.grid(True, which="major", color=GRID, lw=0.8); c.set_axisbelow(True)

fig.savefig("figures/scaling_law.png", dpi=170)
print("saved; product-law fit on thresholds:", np.round(beta, 3))
