"""Kernel measurements from kern.py outputs.
usage: python kern_analyze.py KERN_DIR DATA_DIR OUT_JSON

Per run and kernel (K_std; for Adam also K_adam and K_live), at every checkpoint:
  norm         ||K||_F
  chg_ref      ||K_t - K_ref|| / ||K_ref||          (raw change from the reference checkpoint)
  shp_ref      ||K_t/||K_t|| - K_ref/||K_ref|| ||   (change in shape, scale removed)
  spd          ||K_t - K_prev|| / ||K_prev|| / (epochs between)   (raw change per epoch)
  shp_spd      shape change per epoch
  align        centered alignment of the 250x250 train block with the labels (as sum_rich.py)
  krr          test accuracy of kernel regression, 250 train -> 250 test (as sum_rich.py)
Reference checkpoint: epoch 0 for K_std; the first checkpoint after training starts (epoch 1) for K_adam / K_live,
which are undefined at step 0.
Per-run scalars:
  settle90 / settle90_shape   first epoch where chg_ref (shp_ref) reaches 90% of its final value
  path_ratio / path_ratio_shape   sum of consecutive changes / net change (reference to final); 1 = straight path
  align_tv_ratio, krr_tv_ratio    total variation of the curve / |net change|; 1 = monotone
  align_reversals, krr_reversals  number of direction changes (steps smaller than 0.002 / 0.004 ignored)
  late_spd, late_shp_spd           mean per-epoch change over the last 100 epochs
plus Hessian top eigenvalue curve and train/test accuracy curves."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1_cifar_mlp"))  # core.py
import core

KD, DATA, OUT = sys.argv[1:4]
_, ytr_all, _, yte_all = core.load_cifar(DATA)


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


def reversals(x, tol):
    d = np.diff(np.asarray(x, float))
    d = d[np.abs(d) > tol]
    return int((np.sign(d[1:]) != np.sign(d[:-1])).sum()) if len(d) > 1 else 0


def tv_ratio(x):
    x = np.asarray(x, float)
    net = abs(x[-1] - x[0])
    return float(np.abs(np.diff(x)).sum() / net) if net > 0 else None


def settle(epochs, c, frac=0.9):
    c = np.asarray(c, float)
    if not np.isfinite(c[-1]) or c[-1] <= 0:
        return None
    idx = np.where(c >= frac * c[-1])[0]
    return float(epochs[idx[0]]) if len(idx) else None


runs = {}
for rdir in sorted(glob.glob(os.path.join(KD, "kern_*"))):
    if not os.path.isdir(rdir) or not os.path.exists(os.path.join(rdir, "meta.json")):
        continue
    meta = json.load(open(os.path.join(rdir, "meta.json")))
    ytr_p = ytr_all[np.array(meta["probe_ntk_train"])]
    yte_p = yte_all[np.array(meta["probe_ntk_test"])]
    Ytr = np.eye(10)[ytr_p]
    cks = {c["step"]: c for c in meta["ckpts"] if "kernels" in c}
    files = sorted(glob.glob(os.path.join(rdir, "ck_*.npz")))
    data = {}
    for f in files:
        step = int(os.path.basename(f)[3:10])
        if step not in cks:
            continue
        z = np.load(f)
        for kn in z.files:
            data.setdefault(kn, []).append((cks[step]["epoch"], z[kn].astype(np.float64)))
    res = dict(name=meta["name"], opt=meta["opt"], depth=meta["depth"], seed=meta["seed"], finished=meta.get("finished"),
               diverged=meta.get("diverged"), seconds=meta.get("seconds"), errors=meta.get("errors"),
               curves=dict(epoch=[c["epoch"] for c in meta["ckpts"]], train_acc=[c["train_acc"] for c in meta["ckpts"]],
                           test_acc=[c["test_acc"] for c in meta["ckpts"]],
                           train_loss=[c["train_loss"] for c in meta["ckpts"]],
                           frac_sqrtv_below_eps=[c.get("frac_sqrtv_below_eps") for c in meta["ckpts"]],
                           dead_frac=[c.get("dead_frac") for c in meta["ckpts"]]),
               hessian=meta.get("hessian"), kernels={})
    for kn, lst in data.items():
        lst.sort(key=lambda t: t[0])
        ep = np.array([t[0] for t in lst])
        Ks = [t[1] for t in lst]
        nrm = np.array([np.linalg.norm(K) for K in Ks])
        Kn = [K / n for K, n in zip(Ks, nrm)]
        K0, n0, Kn0 = Ks[0], nrm[0], Kn[0]
        chg = np.array([np.linalg.norm(K - K0) / n0 for K in Ks])
        shp = np.array([np.linalg.norm(k - Kn0) for k in Kn])
        de = np.diff(ep)
        steps_raw = np.array([np.linalg.norm(Ks[i] - Ks[i - 1]) for i in range(1, len(Ks))])
        steps_shp = np.array([np.linalg.norm(Kn[i] - Kn[i - 1]) for i in range(1, len(Ks))])
        spd = steps_raw / nrm[:-1] / de
        shp_spd = steps_shp / de
        al = [centered_alignment(K[:250, :250], Ytr) for K in Ks]
        kr = [krr_acc(K, ytr_p, yte_p, 250) for K in Ks]
        net_raw = np.linalg.norm(Ks[-1] - K0)
        net_shp = np.linalg.norm(Kn[-1] - Kn0)
        late = ep[1:] > ep[-1] - 100
        r = dict(epoch=ep.tolist(), norm=nrm.tolist(), chg_ref=chg.tolist(), shp_ref=shp.tolist(),
                 spd=[None] + spd.tolist(), shp_spd=[None] + shp_spd.tolist(), align=al, krr=kr,
                 settle90=settle(ep, chg), settle90_shape=settle(ep, shp),
                 path_ratio=float(steps_raw.sum() / net_raw) if net_raw > 0 else None,
                 path_ratio_shape=float(steps_shp.sum() / net_shp) if net_shp > 0 else None,
                 align_tv_ratio=tv_ratio(al), krr_tv_ratio=tv_ratio(kr),
                 align_reversals=reversals(al, 0.002), krr_reversals=reversals(kr, 0.004),
                 late_spd=float(spd[late].mean()) if late.any() else None,
                 late_shp_spd=float(shp_spd[late].mean()) if late.any() else None, n_ckpts=len(ep))
        res["kernels"][kn] = r
    runs[meta["name"]] = res

# aggregate over seeds per (opt, depth)
agg = {}
for r in runs.values():
    agg.setdefault(f"{r['opt']}_L{r['depth']}", []).append(r)
summary = {}
for key, rs in sorted(agg.items()):
    s = dict(n=len(rs), finished=[r["finished"] for r in rs], seeds=[r["seed"] for r in rs],
             final_test_acc=[r["curves"]["test_acc"][-1] if r["curves"]["test_acc"] else None for r in rs],
             final_train_acc=[r["curves"]["train_acc"][-1] if r["curves"]["train_acc"] else None for r in rs],
             final_epoch=[r["curves"]["epoch"][-1] if r["curves"]["epoch"] else None for r in rs],
             hessian_final=[r["hessian"][-1]["top"] if r["hessian"] else None for r in rs], kernels={})
    for kn in ("K_std", "K_adam", "K_live"):
        ks = [r["kernels"][kn] for r in rs if kn in r["kernels"]]
        if not ks:
            continue
        s["kernels"][kn] = {m: [k[m] for k in ks] for m in
                            ("settle90", "settle90_shape", "path_ratio", "path_ratio_shape", "align_tv_ratio",
                             "krr_tv_ratio", "align_reversals", "krr_reversals", "late_spd", "late_shp_spd", "n_ckpts")}
        s["kernels"][kn]["final_chg_ref"] = [k["chg_ref"][-1] for k in ks]
        s["kernels"][kn]["final_shp_ref"] = [k["shp_ref"][-1] for k in ks]
        s["kernels"][kn]["final_align"] = [k["align"][-1] for k in ks]
        s["kernels"][kn]["final_krr"] = [k["krr"][-1] for k in ks]
    summary[key] = s
json.dump(dict(runs=runs, summary=summary), open(OUT, "w"))


def fmt(v):
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


for key, s in summary.items():
    print("==", key, "n", s["n"], "finished", s["finished"], "final_epoch", fmt(s["final_epoch"]),
          "test", fmt(s["final_test_acc"]), "hess", fmt(s["hessian_final"]))
    for kn, d in s["kernels"].items():
        print("  ", kn, " ".join(f"{m}={fmt(v)}" for m, v in d.items()))
print("wrote", OUT, os.path.getsize(OUT), "bytes")
