"""Compact extract of rich_summary.json + post.json for plotting/reporting. usage: compact.py SUMM_DIR"""
import json, os, sys
import numpy as np
D = sys.argv[1]
s = json.load(open(os.path.join(D, "rich_summary.json")))


def r3(x):
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, list):
        return [r3(v) for v in x]
    return float(f"{x:.3g}")


out = {"configs": {}, "data_order": {}}
for name, c in s["configs"].items():
    ep = np.array(c["ckpt_epoch"])
    idx = sorted(set(np.searchsorted(ep, np.geomspace(max(ep[1], 1e-3), ep[-1], 22)).tolist() + [0, len(ep) - 1]))
    idx = [i for i in idx if i < len(ep)]
    d = {"n": c["n_seeds"], "ep": r3(ep[idx].tolist()),
         "tr_acc": r3(np.array(c["train_acc_mean"])[idx].tolist()), "te_acc": r3(np.array(c["test_acc_mean"])[idx].tolist()),
         "tr_loss": r3(np.array(c["train_loss_mean"])[idx].tolist()),
         "tr_acc_final": r3(c["train_acc_per_seed_final"]), "tr_acc_max": r3(c["train_acc_per_seed_max"]),
         "te_acc_final": r3(c["test_acc_per_seed_final"]), "te_acc_max": r3(c["test_acc_per_seed_max"]),
         "collapse": c["collapse_epoch_per_seed"], "seedavg_te_final": r3(c["seedavg_test_acc"][-1]),
         "seedavg_te_max": r3(max(c["seedavg_test_acc"]))}
    if "pairwise_test_pred_agreement" in c:
        d["agree_final"] = r3(c["pairwise_test_pred_agreement"][-1])
        d["lt_spearman"] = c.get("learning_time_spearman_between_seeds")
    d["deep_ep"] = r3(c["deep_epoch"])
    for k in ("sharpness_top_mean", "precond_sharpness_top_mean", "ntk_label_alignment_mean",
              "ntk_rel_change_from_init_mean", "ntk_regression_acc_250train_250test_mean"):
        if k in c:
            d[k] = r3(c[k])
    if "adam_frac_coords_sqrtv_below_eps_x1_x10_x100_mean" in c:
        d["eps_frac_last"] = r3(c["adam_frac_coords_sqrtv_below_eps_x1_x10_x100_mean"][-1])
        d["eps_frac_x1"] = r3([x[0] for x in c["adam_frac_coords_sqrtv_below_eps_x1_x10_x100_mean"]])
    for k in ("rep_eff_rank_by_layer_mean", "rep_dead_frac_by_layer_mean", "w_dist_init_by_layer_mean"):
        if k in c:
            d[k.replace("_by_layer_mean", "_last")] = r3(c[k][-1])
    out["configs"][name] = d
for name, v in s.get("data_order", {}).items():
    out["data_order"][name] = {"runs": v["runs"], "pairs": {k: [(p["runs"], p["final_test_pred_agreement"],
                                                                 p["final_test_output_rel_dist"]) for p in ps]
                                                             for k, ps in v["pairs"].items()}}
pf = os.path.join(D, "post.json")
if os.path.exists(pf):
    p = json.load(open(pf))
    po = {}
    for sec in ("seed_pairs", "adam_vs_sgd_same_init", "init_to_final", "epoch150_to_final"):
        po[sec] = {k: {"barrier": r3(v["train_loss_barrier"]), "acc_dip": r3(v["test_acc_dip"]),
                       "tr_loss": r3(v["train_loss"]), "te_acc": r3(v["test_acc"]),
                       "dist": r3(sum(x ** 2 for x in v.get("dist_by_layer", [0])) ** 0.5)} for k, v in p[sec].items()}
    po["hessian_top10"] = {k: r3(v) for k, v in p["hessian_top10"].items()}
    out["post"] = po
print(json.dumps(out, separators=(",", ":")))
