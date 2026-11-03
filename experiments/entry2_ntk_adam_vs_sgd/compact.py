"""Compact extract of kern_analyze.py's output: every per-run scalar, plus the curves at 20 epochs
(0, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, ..., 500), values rounded to 4 significant figures.
usage: python compact.py RESULTS_JSON OUT_JSON"""
import json
import sys

R = json.load(open(sys.argv[1]))
keep = [0, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
out = {}
for nm, r in R["runs"].items():
    e = {"finished": r["finished"], "seconds": r["seconds"], "errors": r["errors"],
         "curves": {k: [v[i] for i, ep in enumerate(r["curves"]["epoch"]) if round(ep) in keep]
                    for k, v in r["curves"].items() if k != "dead_frac"},
         "dead_frac": [r["curves"]["dead_frac"][i] for i, ep in enumerate(r["curves"]["epoch"]) if round(ep) in keep],
         "hessian": [[h["epoch"], round(h["top"], 4), round(h["dominant"], 4)] for h in (r["hessian"] or [])],
         "kernels": {}}
    for kn, k in r["kernels"].items():
        idx = [i for i, ep in enumerate(k["epoch"]) if round(ep) in keep]
        e["kernels"][kn] = {m: k[m] for m in k if not isinstance(k[m], list)}
        for m in ("epoch", "norm", "chg_ref", "shp_ref", "spd", "shp_spd", "align", "krr"):
            e["kernels"][kn][m] = [None if k[m][i] is None else float(f"{k[m][i]:.4g}") for i in idx]
    out[nm] = e
json.dump({"runs": out}, open(sys.argv[2], "w"), separators=(",", ":"))
print("wrote", sys.argv[2], len(out), "runs")
