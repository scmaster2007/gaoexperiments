"""Summaries of the exploratory runs (rich.py outputs).
usage: python sum_rich.py RICH_DIR OUT_JSON"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

R, OUT = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(R, "rich_*.json")))


def is_dataorder(m):
    """run whose data-order seed differs from its init seed"""
    return m.get("data_seed", -1) not in (-1, m["seed"])


def key_of(m):
    return (m["opt"], m["depth"], m["width"], m["batch"], m["lr"], m["eps"], m["sched"], m["epochs"],
            "dataorder" if is_dataorder(m) else "")


def kname(k):
    opt, L, w, b, lr, eps, sched, ep, tag = k
    return f"{opt}_L{L}_w{w}_b{b}_lr{lr:g}_eps{eps:g}_{sched}{ep}" + (f"_{tag}" if tag else "")


def nparams(L, w):
    """numel of each parameter tensor in model.parameters() order: W0,b0,W1,b1,...,Wout,bout"""
    dims = [3072] + [w] * L + [10]
    out = []
    for i in range(L + 1):
        out += [dims[i + 1] * dims[i], dims[i + 1]]
    return np.array(out, float)


def learning_times(bits, n):
    """bits [C, ceil(n/8)] packed correctness at checkpoints -> index of first checkpoint after which the
    example stays correct (C = never)."""
    corr = np.unpackbits(bits, axis=1)[:, :n].astype(bool)  # [C, n]
    C = corr.shape[0]
    stays = np.flip(np.cumprod(np.flip(corr, 0), 0), 0).astype(bool)  # stays[c] = correct from c to end
    first = np.where(stays.any(0), stays.argmax(0), C)
    return first, corr


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def centered_alignment(K, Y):
    n = len(K)
    H = np.eye(n) - 1.0 / n
    Kc, Yc = H @ K @ H, H @ (Y @ Y.T) @ H
    return float((Kc * Yc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Yc) + 1e-300))


def krr_acc(K, ytr, yte, ntr, ridge=1e-6):
    Ktt, Kst = K[:ntr, :ntr], K[ntr:, :ntr]
    Y = np.eye(10)[ytr]
    a = np.linalg.solve(Ktt + ridge * np.trace(Ktt) / ntr * np.eye(ntr), Y)
    return float(((Kst @ a).argmax(1) == yte).mean())


groups = defaultdict(list)
for f in files:
    m = json.load(open(f))
    groups[key_of(m)].append((m, f[:-5] + ".npz"))

# labels for probes (CIFAR-10 order identical on every node)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
_, ytr_all, _, yte_all = core.load_cifar(os.path.expanduser(os.environ.get("CIFAR", "~/data")))
if os.environ.get("SMALL"):  # local test mode, matches rich.py --small
    ytr_all, yte_all = ytr_all[:2000], yte_all[:500]

summary = {"configs": {}}
for k, runs in sorted(groups.items()):
    runs.sort(key=lambda r: r[0]["seed"])
    ms = [r[0] for r in runs]
    zs = [np.load(r[1]) for r in runs]
    L, w = k[1], k[2]
    c = {"n_seeds": len(ms), "seeds": [m["seed"] for m in ms],
         "data_seeds": [m["seed"] if m.get("data_seed", -1) < 0 else m["data_seed"] for m in ms],
         "diverged": [m["diverged"] for m in ms],
         "seconds": [round(m["seconds"]) for m in ms], "params": ms[0]["params"]}
    # curves (all seeds share the checkpoint schedule unless a run diverged early)
    nck = min(len(m["curves"]["step"]) for m in ms)
    ep = np.array(ms[0]["curves"]["epoch"][:nck])
    c["ckpt_epoch"] = ep.round(4).tolist()
    for q in ("train_loss", "train_acc", "test_loss", "test_acc"):
        A = np.array([m["curves"][q][:nck] for m in ms], float)
        c[q + "_mean"] = np.nanmean(A, 0).round(5).tolist()
        c[q + "_per_seed_final"] = [round(float(m["curves"][q][-1]), 5) for m in ms]
        c[q + "_per_seed_max"] = [round(float(np.nanmax(m["curves"][q])), 5) for m in ms]
    # collapse: after reaching train_acc >= 0.8, first checkpoint with train_acc < 0.2
    col = []
    for m in ms:
        ta = np.array(m["curves"]["train_acc"]); e = np.array(m["curves"]["epoch"])
        hit = np.where(ta >= 0.8)[0]
        if len(hit):
            after = np.where((ta < 0.2) & (np.arange(len(ta)) > hit[0]))[0]
            col.append(float(e[after[0]]) if len(after) else None)
        else:
            col.append("never reached 0.8")
    c["collapse_epoch_per_seed"] = col
    # seed-averaged function on the test set
    T = np.stack([z["test_out"][:nck].astype(np.float32) for z in zs])  # [S, C, 10000, 10]
    avg = T.mean(0)
    c["seedavg_test_acc"] = (avg.argmax(-1) == yte_all[None]).mean(1).round(5).tolist()
    if len(ms) > 1:
        pred = T.argmax(-1)
        agree = []
        for ci in range(nck):
            pa = [(pred[i, ci] == pred[j, ci]).mean() for i in range(len(ms)) for j in range(i + 1, len(ms))]
            agree.append(float(np.mean(pa)))
        c["pairwise_test_pred_agreement"] = np.round(agree, 5).tolist()
        rel = [float(np.mean([np.linalg.norm(T[s, ci] - avg[ci]) / (np.linalg.norm(avg[ci]) + 1e-12)
                              for s in range(len(ms))])) for ci in range(nck)]
        c["mean_rel_dist_seed_to_seedavg"] = np.round(rel, 5).tolist()
    # per-example learning times on the training set
    lts = []
    never = []
    for z in zs:
        first, _ = learning_times(z["train_correct_bits"][:nck], len(ytr_all))
        lts.append(first); never.append(float((first == nck).mean()))
    c["frac_train_never_learned"] = np.round(never, 5).tolist()
    if len(ms) > 1:
        c["learning_time_spearman_between_seeds"] = round(float(np.mean(
            [spearman(lts[i], lts[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))])), 4)
    # deep diagnostics (mean over seeds at shared deep steps)
    nd = min(len(z["deep_step"]) for z in zs)
    spe = ms[0]["steps_per_epoch"]
    c["deep_epoch"] = (np.array(zs[0]["deep_step"][:nd]) / spe).round(3).tolist()

    def dmean(key, idx=None):
        arrs = []
        for z in zs:
            if "deep_" + key not in z.files:
                return None
            a = z["deep_" + key]
            arrs.append(a[:nd] if idx is None else a[:nd, idx])
        return np.nanmean(np.stack(arrs), 0)

    sh = dmean("sharpness", 0)
    if sh is not None:
        c["sharpness_top_mean"] = sh.round(4).tolist()
        c["sharpness_top_per_seed"] = [z["deep_sharpness"][:nd, 0].round(4).tolist() for z in zs]
    if k[0] in ("adam", "adamw"):
        ps = dmean("precond_sharpness", 0)
        if ps is not None:
            c["precond_sharpness_top_mean"] = ps.round(2).tolist()
            c["precond_sharpness_top_per_seed"] = [z["deep_precond_sharpness"][:nd, 0].round(2).tolist() for z in zs]
        c["adam_threshold_38_over_lr"] = 38.0 / k[4]
        # eps dominance (fraction of all coordinates, weighted by tensor size); the Adam arrays start at deep step 1
        npar = nparams(L, w)
        fr = []
        for z in zs:
            if "deep_adam_frac_sqrtv_below_eps_x1_10_100" in z.files:
                a = z["deep_adam_frac_sqrtv_below_eps_x1_10_100"]  # [D-1, P, 3]
                fr.append((a * npar[None, :, None]).sum(1) / npar.sum())
        if fr:
            nm = min(len(x) for x in fr)
            c["adam_frac_coords_sqrtv_below_eps_x1_x10_x100_mean"] = np.mean([x[:nm] for x in fr], 0).round(6).tolist()
            c["adam_deep_epochs_for_eps_stats"] = c["deep_epoch"][1:1 + nm]
            rq = np.mean([z["deep_adam_ratio_quantiles"][:nm] for z in zs], 0)  # [D-1, P, 9]
            c["adam_update_ratio_median_by_param_last"] = rq[-1, :, 4].round(4).tolist()
    for key in ("w_fro", "w_maxabs", "w_dist_init", "rep_eff_rank", "rep_cka_labels", "rep_dead_frac"):
        v = dmean(key)
        if v is not None:
            c[key + "_by_layer_mean"] = v.round(4).tolist()
    # per-neuron balance: in_sq - out_sq
    bal_mean, bal_std, bal_corr = [], [], []
    for z in zs:
        if "deep_neuron_in_sq" in z.files:
            a, b = z["deep_neuron_in_sq"][:nd], z["deep_neuron_out_sq"][:nd]  # [D, L, w]
            d = a - b
            bal_mean.append(d.mean(-1)); bal_std.append(d.std(-1))
            bal_corr.append(np.array([[np.corrcoef(a[i, l], b[i, l])[0, 1] for l in range(a.shape[1])]
                                      for i in range(a.shape[0])]))
    if bal_mean:
        c["neuron_balance_in_minus_out_mean_by_layer"] = np.mean(bal_mean, 0).round(5).tolist()
        c["neuron_balance_in_minus_out_std_by_layer"] = np.mean(bal_std, 0).round(5).tolist()
        c["neuron_in_out_corr_by_layer"] = np.mean(bal_corr, 0).round(4).tolist()
    # NTK on the probe points
    al, chg, kacc = [], [], []
    for z in zs:
        if "deep_ntk" not in z.files:
            continue
        K = z["deep_ntk"][:nd].astype(np.float64)
        ytr_p = ytr_all[z["probe_ntk_train"]]; yte_p = yte_all[z["probe_ntk_test"]]
        Ytr = np.eye(10)[ytr_p]
        al.append([centered_alignment(Ki[:250, :250], Ytr) for Ki in K])
        chg.append([float(np.linalg.norm(Ki - K[0]) / np.linalg.norm(K[0])) for Ki in K])
        kacc.append([krr_acc(Ki, ytr_p, yte_p, 250) for Ki in K])
    if al:
        c["ntk_label_alignment_mean"] = np.mean(al, 0).round(4).tolist()
        c["ntk_rel_change_from_init_mean"] = np.mean(chg, 0).round(4).tolist()
        c["ntk_regression_acc_250train_250test_mean"] = np.mean(kacc, 0).round(4).tolist()
    summary["configs"][kname(k)] = c

# init seed vs data-order seed: compare runs of the same config pairwise, grouped by what they share
summary["data_order"] = {}
for k in sorted(groups):
    if k[-1] != "dataorder" or k[:-1] + ("",) not in groups:
        continue
    runs = groups[k[:-1] + ("",)] + groups[k]
    ids, outs, info = [], [], {}
    for m, npz in runs:
        d = m["seed"] if m.get("data_seed", -1) < 0 else m["data_seed"]
        z = np.load(npz)
        ids.append((m["seed"], d)); outs.append(z["test_out"].astype(np.float32))
        ta = np.array(m["curves"]["train_acc"]); e = np.array(m["curves"]["epoch"])
        hit = np.where(ta >= 0.8)[0]
        col = None if not len(hit) else next((float(e[i]) for i in range(hit[0] + 1, len(ta)) if ta[i] < 0.2), None)
        info[f"init{m['seed']}_data{d}"] = {"final_test_acc": round(float(m["curves"]["test_acc"][-1]), 5),
                                            "max_train_acc": round(float(np.nanmax(ta)), 5),
                                            "collapse_epoch": col if len(hit) else "never reached 0.8"}
    nck = min(len(o) for o in outs)
    ep = groups[k[:-1] + ("",)][0][0]["curves"]["epoch"][:nck]
    pairs = {"same_init_diff_data": [], "same_data_diff_init": [], "diff_both": []}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (si, di), (sj, dj) = ids[i], ids[j]
            if (si, di) == (sj, dj):
                continue
            kind = "same_init_diff_data" if si == sj else "same_data_diff_init" if di == dj else "diff_both"
            A, B = outs[i][:nck], outs[j][:nck]
            agree = (A.argmax(-1) == B.argmax(-1)).mean(1)
            dist = np.linalg.norm((A - B).reshape(nck, -1), axis=1) / (
                0.5 * (np.linalg.norm(A.reshape(nck, -1), axis=1) + np.linalg.norm(B.reshape(nck, -1), axis=1)) + 1e-12)
            pairs[kind].append({"runs": [f"init{si}_data{di}", f"init{sj}_data{dj}"],
                                "final_test_pred_agreement": round(float(agree[-1]), 5),
                                "final_test_output_rel_dist": round(float(dist[-1]), 5),
                                "test_pred_agreement_curve": agree.round(4).tolist()})
    summary["data_order"][kname(k[:-1] + ("",))] = {"ckpt_epoch": np.round(ep, 4).tolist(), "runs": info,
                                                     "pairs": pairs}

json.dump(summary, open(OUT, "w"))
# compact table to stdout
print(f"{'config':58s} {'n':>2s} {'final train acc':>16s} {'final test acc':>15s} {'seedavg test':>12s} {'collapse epochs'}")
for name, c in summary["configs"].items():
    print(f"{name:58s} {c['n_seeds']:2d} {np.mean(c['train_acc_per_seed_final']):16.4f} "
          f"{np.mean(c['test_acc_per_seed_final']):15.4f} {c['seedavg_test_acc'][-1]:12.4f} {c['collapse_epoch_per_seed']}")
print("\ndata order vs init (final test prediction agreement, mean over pairs)")
for name, d in summary["data_order"].items():
    s = {kind: (round(float(np.mean([p["final_test_pred_agreement"] for p in v])), 4), len(v))
         for kind, v in d["pairs"].items() if v}
    print(f"{name:58s} {s}")
    for r, v in d["runs"].items():
        print(f"    {r:14s} {v}")
