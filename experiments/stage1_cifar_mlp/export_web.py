"""Compact data file for the web explorer. usage: export_web.py RICH_DIR SUMM_DIR OUT_JSON
Per-run curves (downsampled), per-config summaries and deep diagnostics, post-training measurements,
Adam-at-width summaries, R1-R4 summaries. Numbers rounded to 3-4 significant figures; NaN/inf -> null."""
import glob, json, math, os, sys
import numpy as np

R, S, OUT = (os.path.expanduser(a) for a in sys.argv[1:4])


def r(x, s=3):
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, dict):
        return {k: r(v, s) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, s) for v in x]
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{s}g}")


def cfg(m):
    return f"{m['opt']}_L{m['depth']}_w{m['width']}_b{m['batch']}_lr{m['lr']:g}_eps{m['eps']:g}_{m['sched']}{m['epochs']}"


targets = list(np.geomspace(0.02, 10, 10)) + list(range(20, 501, 20))


def pick(ep):
    ep = np.asarray(ep, float)
    idx = {0, len(ep) - 1}
    for t in targets:
        if t <= ep[-1] + 1e-9:
            idx.add(int(np.argmin(np.abs(ep - t))))
    return sorted(idx)


out = {"runs": [], "configs": {}}
for f in sorted(glob.glob(os.path.join(R, "rich_*.json"))):
    m = json.load(open(f))
    c = m["curves"]
    idx = pick(c["epoch"])
    ds = m.get("data_seed", -1)
    out["runs"].append({
        "cfg": cfg(m), "seed": m["seed"], "ds": m["seed"] if ds is None or ds < 0 else ds,
        "e": r([c["epoch"][i] for i in idx]),
        "tra": [round(1000 * c["train_acc"][i]) if c["train_acc"][i] == c["train_acc"][i] else None for i in idx],
        "tea": [round(1000 * c["test_acc"][i]) if c["test_acc"][i] == c["test_acc"][i] else None for i in idx],
        "trl": r([c["train_loss"][i] for i in idx]),
    })

rs = json.load(open(os.path.join(S, "rich_summary.json")))
for name, c in rs["configs"].items():
    idx = pick(c["ckpt_epoch"])
    d = {"n": c["n_seeds"], "seeds": c["seeds"], "data_seeds": c.get("data_seeds"),
         "collapse": c["collapse_epoch_per_seed"],
         "tra_f": r(c["train_acc_per_seed_final"]), "tra_max": r(c["train_acc_per_seed_max"]),
         "tea_f": r(c["test_acc_per_seed_final"]), "tea_max": r(c["test_acc_per_seed_max"]),
         "never": r(c.get("frac_train_never_learned")), "lt_rho": c.get("learning_time_spearman_between_seeds"),
         "ce": r([c["ckpt_epoch"][i] for i in idx]),
         "savg": r([c["seedavg_test_acc"][i] for i in idx])}
    if "pairwise_test_pred_agreement" in c:
        d["agree"] = r([c["pairwise_test_pred_agreement"][i] for i in idx])
        d["reldist"] = r([c["mean_rel_dist_seed_to_seedavg"][i] for i in idx])
    d["de"] = r(c["deep_epoch"])
    for k, kk in (("sharpness_top_mean", "sharp"), ("precond_sharpness_top_mean", "psharp"),
                  ("ntk_label_alignment_mean", "ntk_align"), ("ntk_rel_change_from_init_mean", "ntk_change"),
                  ("ntk_regression_acc_250train_250test_mean", "ntk_acc"),
                  ("rep_dead_frac_by_layer_mean", "dead"), ("rep_eff_rank_by_layer_mean", "erank"),
                  ("rep_cka_labels_by_layer_mean", "cka"), ("w_dist_init_by_layer_mean", "wdist"),
                  ("w_fro_by_layer_mean", "wfro")):
        if k in c:
            d[kk] = r(c[k])
    if "adam_frac_coords_sqrtv_below_eps_x1_x10_x100_mean" in c:
        d["epsf"] = r([x[0] for x in c["adam_frac_coords_sqrtv_below_eps_x1_x10_x100_mean"]])
        d["epsf10"] = r([x[1] for x in c["adam_frac_coords_sqrtv_below_eps_x1_x10_x100_mean"]])
        d["epse"] = r(c["adam_deep_epochs_for_eps_stats"])
    out["configs"][name] = d
out["data_order"] = {}
for name, v in rs.get("data_order", {}).items():
    out["data_order"][name] = {"runs": v["runs"], "pairs": {k: [[p["runs"], p["final_test_pred_agreement"],
                                                                 p["final_test_output_rel_dist"]] for p in ps]
                                                             for k, ps in v["pairs"].items()}}

pf = os.path.join(S, "post.json")
if os.path.exists(pf):
    p = json.load(open(pf))
    po = {}
    for sec in ("seed_pairs", "adam_vs_sgd_same_init", "init_to_final", "epoch150_to_final"):
        po[sec] = {k: {"b": r(v["train_loss_barrier"]), "dip": r(v["test_acc_dip"]), "trl": r(v["train_loss"]),
                       "tea": r(v["test_acc"]),
                       "dist": r(math.sqrt(sum(x * x for x in v["dist_by_layer"]))) if "dist_by_layer" in v else None,
                       "dl": r(v.get("dist_by_layer"))} for k, v in p.get(sec, {}).items()}
    po["hess"] = {k: (r(v) if not isinstance(v, str) else None) for k, v in p.get("hessian_top10", {}).items()}
    out["post"] = po

af = os.path.join(S, "adamwide_summary.json")
if os.path.exists(af):
    out["adamwide"] = r(json.load(open(af)), 4)

sf = os.path.join(S, "repro_figs", "summary.json")
if os.path.exists(sf):
    s = json.load(open(sf))
    rep = {}
    if "r12" in s:
        rep["r12"] = {k: v for k, v in r(s["r12"], 4).items()}
    for k in ("r3sig", "r3train", "r4full", "pilot"):
        if k in s:
            rep[k] = r(s[k], 4)
    for k in ("r4_n5000", "r4_n20000"):
        if k in s:
            v = s[k]
            rep[k] = r({kk: v[kk] for kk in ("n_train", "n_test", "depths", "ntk_best_acc", "nngp_best_acc",
                                            "ntk_offdiag_over_diag", "nngp_offdiag_over_diag", "ntk_diag", "nngp_diag",
                                            "ntk_pred_input_std", "nngp_pred_input_std", "ntk_kappa") if kk in v}, 4)
    out["repro"] = rep

json.dump(out, open(OUT, "w"), separators=(",", ":"))
print(len(out["runs"]), "runs,", len(out["configs"]), "configs,", os.path.getsize(OUT), "bytes")
