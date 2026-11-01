"""Summaries and figures for the reproductions.
usage: python analyze.py RESULTS_DIR FIG_DIR"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = sys.argv[1], sys.argv[2]
os.makedirs(F, exist_ok=True)
summary = {}


def slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


# ----------------------------------------------------------------------------- R1 + R2
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/r12_*.json"))]
if rows:
    by = defaultdict(list)
    for r in rows:
        by[(r["depth"], r["width"])].append(r)
    depths = sorted({r["depth"] for r in rows})
    keys = [("init_kernel_rel_err", "R1: ||K_emp(init) - K_inf|| / ||K_inf||"),
            ("weight_rel_change", "R2: ||theta_T - theta_0|| / ||theta_0||"),
            ("kernel_rel_change", "R2: ||K_T - K_0|| / ||K_0||"),
            ("test_rel_dist_net_vs_lin0", "R2: test outputs, network vs own linearization"),
            ("test_rel_dist_net_vs_lininf", "R2: test outputs, network vs infinite-width linear model")]
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(4.2 * (len(keys) + 1), 3.6))
    s2 = {}
    # seed average vs infinite-width mean predictor
    for L in depths:
        ws = sorted({w for (d, w) in by if d == L})
        for k, (key, title) in enumerate(keys):
            m = [np.mean([r[key] for r in by[(L, w)]]) for w in ws]
            sd = [np.std([r[key] for r in by[(L, w)]]) for w in ws]
            axes[k].errorbar(ws, m, yerr=sd, marker="o", ms=3, label=f"{L} hidden layers")
            s2.setdefault(key, {})[L] = {"widths": ws, "mean": m, "sd": sd, "n_seeds": [len(by[(L, w)]) for w in ws],
                                        "loglog_slope": slope(ws, m) if len(ws) > 1 else None}
        dist = []
        for w in ws:
            fs = []
            for r in by[(L, w)]:
                z = np.load(f"{R}/r12_L{L}_w{w}_s{r['seed']}.npz")
                n = r["n_train"]
                fs.append(z["traj"][-1][n:])
                mean_inf = z["meaninf"][-1][n:]
            avg = np.mean(fs, 0)
            dist.append(float(np.linalg.norm(avg - mean_inf) / np.linalg.norm(mean_inf)))
        axes[-1].plot(ws, dist, marker="o", ms=3, label=f"{L} hidden layers")
        s2.setdefault("seed_avg_vs_inf_mean_predictor", {})[L] = {"widths": ws, "rel_dist": dist,
                                                                   "loglog_slope": slope(ws, dist) if len(ws) > 1 else None}
    for k, (key, title) in enumerate(keys):
        axes[k].set(xscale="log", yscale="log", xlabel="width", title=title)
        axes[k].title.set_fontsize(8)
    axes[-1].set(xscale="log", yscale="log", xlabel="width",
                 title="R2: seed-averaged test outputs vs infinite-width mean prediction")
    axes[-1].title.set_fontsize(8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{F}/r12.png", dpi=130)
    # agreement and accuracy table at the largest width
    tab = {}
    for (L, w), rs in sorted(by.items()):
        tab[f"L{L}_w{w}"] = {k: float(np.mean([r[k] for r in rs])) for k in
                             ("test_agree_net_vs_lin0", "test_agree_net_vs_lininf", "test_acc_net",
                              "test_acc_lininf", "test_acc_meaninf")}
        tab[f"L{L}_w{w}"]["final_train_loss"] = float(np.mean([r["train_loss"][str(r["steps"])] for r in rs]))
        tab[f"L{L}_w{w}"]["n_seeds"] = len(rs)
    s2["table"] = tab
    summary["r12"] = s2

# ----------------------------------------------------------------------------- R3a
for f in glob.glob(f"{R}/r3sig_L*.json"):
    d = json.load(open(f))
    L = d["depth"]
    res = d["results"]
    q0 = d["x_sq_over_d"]
    l = np.arange(1, L + 1)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    s3 = {}
    for sw2 in sorted({r["sw2"] for r in res}):
        rs = [r for r in res if r["sw2"] == sw2 and r["width"] == 4096]
        q = np.mean([r["q"] for r in rs], 0)
        g = np.mean([r["g"] for r in rs], 0)
        q_th = sw2 * q0 * (sw2 / 2) ** (l - 1)
        g_th = 0.5 * (sw2 / 2) ** (L - l)
        ax[0].plot(l, q, label=f"{sw2}")
        ax[0].plot(l, q_th, "k:", lw=0.6)
        ax[2].plot(l, g)
        ax[2].plot(l, g_th, "k:", lw=0.6)
        s3[sw2] = {"q_ratio_emp_over_theory_at_layers_1_10_L": [float(q[0] / q_th[0]), float(q[9] / q_th[9]),
                                                               float(q[-1] / q_th[-1])],
                   "g_ratio_emp_over_theory_at_layers_1_10_L": [float(g[0] / g_th[0]), float(g[9] / g_th[9]),
                                                               float(g[-1] / g_th[-1])],
                   "q_last_over_q_first": float(q[-1] / q[0])}
    c = np.mean([r["c"] for r in res if r["sw2"] == 2.0 and r["width"] == 4096], 0)
    ax[1].plot(l, c, label="measured (width 4096)")
    ax[1].plot(l, d["c_theory"], "k:", label="theory")
    s3["corr_sw2_2_measured_layers_1_10_L"] = [float(c[0]), float(c[9]), float(c[-1])]
    s3["corr_theory_layers_1_10_L"] = [d["c_theory"][0], d["c_theory"][9], d["c_theory"][-1]]
    ax[0].set(yscale="log", xlabel="layer", title="pre-activation variance (dotted: theory)")
    ax[0].legend(title="weight variance x fan-in", fontsize=7)
    ax[1].set(xlabel="layer", title="correlation between two inputs (weight variance 2)")
    ax[1].legend(fontsize=7)
    ax[2].set(yscale="log", xlabel="layer", title="squared gradient norm wrt pre-activations")
    fig.tight_layout()
    fig.savefig(f"{F}/r3sig.png", dpi=130)
    summary["r3sig"] = s3

# ----------------------------------------------------------------------------- R3b
rows = [json.load(open(f)) for f in glob.glob(f"{R}/r3train_*.json")]
if rows:
    depths = sorted({r["depth"] for r in rows})
    sws = sorted({r["sw2"] for r in rows})
    best = np.full((len(depths), len(sws)), np.nan)
    for r in rows:
        i, j = depths.index(r["depth"]), sws.index(r["sw2"])
        a = r["history"][-1]["train_acc"] if r["finite"] else 0.0
        best[i, j] = np.nanmax([best[i, j], a])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(best, vmin=0.1, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(sws)), [str(s) for s in sws])
    ax.set_yticks(range(len(depths)), [str(d) for d in depths])
    for i in range(len(depths)):
        for j in range(len(sws)):
            ax.text(j, i, f"{best[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
    ax.set(xlabel="weight variance x fan-in", ylabel="hidden layers",
           title="R3b: training accuracy after SGD (best learning rate)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{F}/r3train.png", dpi=130)
    summary["r3train"] = {"depths": depths, "sw2": sws, "best_train_acc": best.tolist()}

# ----------------------------------------------------------------------------- R4
for f in glob.glob(f"{R}/r4_n*.json"):
    d = json.load(open(f))
    rows = d["rows"]
    L = np.array([r["depth"] for r in rows])
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    s4 = {"n_train": d["n_train"], "n_test": d["n_test"], "depths": L.tolist()}
    for kname in ("ntk", "nngp"):
        accs = {rr: [r.get(f"{kname}_acc_ridge{rr}") for r in rows] for rr in ("1e-08", "1e-06", "0.0001", "0.01")}
        best_acc = [max([a for a in (accs[rr][i] for rr in accs) if a is not None], default=np.nan) for i in range(len(L))]
        ax[0].plot(L, best_acc, marker="o", ms=3, label=kname)
        s4[f"{kname}_best_acc"] = best_acc
        s4[f"{kname}_acc_by_ridge"] = accs
        diag = np.array([r[f"{kname}_diag_mean"] for r in rows])
        off = np.array([r[f"{kname}_offdiag_mean"] for r in rows])
        ax[1].plot(L, diag, label=f"{kname} diagonal")
        ax[1].plot(L, off, "--", label=f"{kname} off-diagonal")
        s4[f"{kname}_diag"] = diag.tolist()
        s4[f"{kname}_offdiag"] = off.tolist()
        s4[f"{kname}_offdiag_over_diag"] = (off / diag).tolist()
        k12 = [r[f"{kname}_kappa_cifar_m12"] for r in rows]
        k20 = [r[f"{kname}_kappa_cifar_m20"] for r in rows]
        g12 = [r[f"{kname}_kappa_gauss_m12"] for r in rows]
        g20 = [r[f"{kname}_kappa_gauss_m20"] for r in rows]
        if kname == "ntk":
            ax[2].plot(L, k12, label="CIFAR, m=12"); ax[2].plot(L, k20, label="CIFAR, m=20")
            ax[2].plot(L, g12, ":", label="Gaussian, m=12"); ax[2].plot(L, g20, ":", label="Gaussian, m=20")
            ax[2].axhline(15 / 3, color="k", lw=0.5); ax[2].axhline(23 / 3, color="k", lw=0.5)
        s4[f"{kname}_kappa"] = {"cifar_m12": k12, "cifar_m20": k20, "gauss_m12": g12, "gauss_m20": g20}
        std = [r.get(f"{kname}_pred_input_std_ridge1e-06") for r in rows]
        ax[3].plot(L, std, marker="o", ms=3, label=kname)
        deep = L >= 16
        s4[f"{kname}_pred_input_std"] = std
        if deep.sum() > 2 and all(s is not None for s in np.array(std, dtype=object)[deep]):
            s4[f"{kname}_pred_input_std_loglog_slope_depth_ge16"] = slope(L[deep], np.array(std, float)[deep])
            s4[f"{kname}_diag_loglog_slope_depth_ge16"] = slope(L[deep], diag[deep])
    ax[0].set(xscale="log", xlabel="hidden layers", ylabel="test accuracy", title="kernel regression accuracy")
    ax[1].set(xscale="log", yscale="log", xlabel="hidden layers", title="mean kernel entries")
    ax[2].set(xscale="log", yscale="log", xlabel="hidden layers", title="NTK condition number (lines: (m+3)/3)")
    ax[3].set(xscale="log", yscale="log", xlabel="hidden layers", title="spread of predictions across test inputs")
    for a in ax:
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{F}/{os.path.basename(f)[:-5]}.png", dpi=130)
    summary[os.path.basename(f)[:-5]] = s4

if os.path.exists(f"{R}/r4full.json"):
    summary["r4full"] = json.load(open(f"{R}/r4full.json"))

# ----------------------------------------------------------------------------- pilot
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/pilot_*.json"))]
if rows:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    sp = {}
    for r in rows:
        h = r["history"]
        ep = [x["epoch"] for x in h]
        ax[0].plot(ep, [x["running_train_loss"] for x in h], label=f"{r['depth']} hidden, batch {r['batch']}")
        e2 = [x["epoch"] for x in h if "test_acc" in x]
        ax[1].plot(e2, [x["test_acc"] for x in h if "test_acc" in x])
        ax[1].plot(e2, [x["train_acc"] for x in h if "train_acc" in x], ":")
        full = [(x["epoch"], x["train_loss"], x["train_acc"], x["test_acc"]) for x in h if "train_loss" in x]
        sp[f"L{r['depth']}_b{r['batch']}"] = {"last": full[-1],
                                              "every_50": [f for f in full if f[0] % 50 == 0]}
    ax[0].set(yscale="log", xlabel="epoch", ylabel="training loss (running)", title="Adam pilot (PyTorch defaults)")
    ax[1].set(xlabel="epoch", ylabel="accuracy (solid test, dotted train)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{F}/pilot.png", dpi=130)
    summary["pilot"] = sp

json.dump(summary, open(f"{F}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:6000])
