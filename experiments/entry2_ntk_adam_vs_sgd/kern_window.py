"""The kern_analyze.py measurements restricted to checkpoints with epoch <= MAX_EPOCH. Same definitions as kern_analyze.py;
the window's last checkpoint plays the role of "final" (settle90, path ratios, final values), and late_spd /
late_shp_spd average the per-epoch change over the window's last 100 epochs.
usage: python kern_window.py KERN_DIR DATA_DIR MAX_EPOCH OUT_JSON"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1_cifar_mlp"))  # core.py
import core

KD, DATA, MAXEP, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
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


SCAL = ("settle90", "settle90_shape", "path_ratio", "path_ratio_shape", "align_tv_ratio", "krr_tv_ratio",
        "align_reversals", "krr_reversals", "late_spd", "late_shp_spd", "n_ckpts", "last_epoch",
        "final_chg_ref", "final_shp_ref", "final_align", "final_krr")
runs = {}
for rdir in sorted(glob.glob(os.path.join(KD, "kern_*"))):
    if not os.path.isdir(rdir) or not os.path.exists(os.path.join(rdir, "meta.json")):
        continue
    meta = json.load(open(os.path.join(rdir, "meta.json")))
    ytr_p = ytr_all[np.array(meta["probe_ntk_train"])]
    yte_p = yte_all[np.array(meta["probe_ntk_test"])]
    Ytr = np.eye(10)[ytr_p]
    cks = {c["step"]: c for c in meta["ckpts"] if "kernels" in c and c["epoch"] <= MAXEP + 1e-9}
    data = {}
    for f in sorted(glob.glob(os.path.join(rdir, "ck_*.npz"))):
        step = int(os.path.basename(f)[3:10])
        if step not in cks:
            continue
        z = np.load(f)
        for kn in z.files:
            data.setdefault(kn, []).append((cks[step]["epoch"], z[kn].astype(np.float64)))
    ck_in = [c for c in meta["ckpts"] if c["epoch"] <= MAXEP + 1e-9]
    res = dict(opt=meta["opt"], depth=meta["depth"], seed=meta["seed"],
               curves=dict(epoch=[c["epoch"] for c in ck_in], train_acc=[round(c["train_acc"], 4) for c in ck_in],
                           test_acc=[round(c["test_acc"], 4) for c in ck_in],
                           train_loss=[round(c["train_loss"], 5) for c in ck_in]),
               hessian=[[h["epoch"], round(h["top"], 3)] for h in (meta.get("hessian") or []) if h["epoch"] <= MAXEP + 1e-9],
               kernels={})
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
        res["kernels"][kn] = dict(
            settle90=settle(ep, chg), settle90_shape=settle(ep, shp),
            path_ratio=float(steps_raw.sum() / net_raw) if net_raw > 0 else None,
            path_ratio_shape=float(steps_shp.sum() / net_shp) if net_shp > 0 else None,
            align_tv_ratio=tv_ratio(al), krr_tv_ratio=tv_ratio(kr),
            align_reversals=reversals(al, 0.002), krr_reversals=reversals(kr, 0.004),
            late_spd=float(spd[late].mean()) if late.any() else None,
            late_shp_spd=float(shp_spd[late].mean()) if late.any() else None,
            n_ckpts=len(ep), last_epoch=float(ep[-1]), final_chg_ref=float(chg[-1]), final_shp_ref=float(shp[-1]),
            final_align=al[-1], final_krr=kr[-1])
    runs[meta["name"]] = res

summary = {}
for r in runs.values():
    s = summary.setdefault(f"{r['opt']}_L{r['depth']}", {"seeds": [], "kernels": {}})
    s["seeds"].append(r["seed"])
    for kn, k in r["kernels"].items():
        d = s["kernels"].setdefault(kn, {m: [] for m in SCAL})
        for m in SCAL:
            d[m].append(k[m])
out = {"max_epoch": MAXEP, "summary": summary,
       "curves": {n: dict(r["curves"], hessian=r["hessian"]) for n, r in runs.items()
                  if r["opt"] == "adam" and r["depth"] == 8}}
txt = json.dumps(out, separators=(",", ":"))
open(OUT, "w").write(txt)
print("wrote", OUT, len(txt), "bytes,", len(runs), "runs")
