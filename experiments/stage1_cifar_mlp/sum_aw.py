"""Summary of adamwide.py runs (Adam in the NTK parametrization across width).
usage: python sum_aw.py AW_DIR OUT_JSON"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
g = defaultdict(list)
for f in sorted(glob.glob(os.path.join(D, "aw_*.json"))):
    m = json.load(open(f))
    g[(m["depth"], m["width"], m["p"])].append(m)
out = {}
preds = ["krr_ntk_init", "krr_asign_init", "krr_sign_init", "krr_ntk_inf", "krr_ntk_inf_mean"]
for (L, w, p), ms in sorted(g.items()):
    r = {"n_seeds": len(ms), "lr": ms[0]["lr"],
         "final_train_loss": float(np.mean([m["train_loss"][str(m["steps"])] for m in ms])),
         "test_acc_net": float(np.mean([m["test_acc_net"] for m in ms])),
         "weight_rel_change": float(np.mean([m["weight_rel_change"] for m in ms])),
         "init_ntk_rel_err_vs_inf": float(np.mean([m["init_ntk_rel_err_vs_inf"] for m in ms]))}
    for k in ("ntk", "asign", "sign"):
        r[f"kernel_rel_change_{k}"] = float(np.mean([m["kernel_rel_change"][k] for m in ms]))
    for pr in preds:
        r[f"rel_dist_net_vs_{pr}"] = float(np.mean([m["test_rel_dist_net_vs"][pr] for m in ms]))
        r[f"agree_net_vs_{pr}"] = float(np.mean([m["test_agree_net_vs"][pr] for m in ms]))
        r[f"test_acc_{pr}"] = float(np.mean([m["test_acc_pred"][pr] for m in ms]))
    out[f"L{L}_w{w}_p{p:g}"] = r
json.dump(out, open(OUT, "w"), indent=0)
print(f"{'config':16s} {'lr':>8s} {'trainloss':>9s} {'acc':>5s} {'dW':>7s} {'dK_ntk':>7s} | rel dist net vs: "
      f"{'ntk0':>6s} {'asign0':>6s} {'sign0':>6s} {'ntkinf':>6s} | agree: {'ntk0':>5s} {'asign0':>6s} {'sign0':>5s}")
for k, r in out.items():
    print(f"{k:16s} {r['lr']:8.2e} {r['final_train_loss']:9.4f} {r['test_acc_net']:5.3f} {r['weight_rel_change']:7.4f} "
          f"{r['kernel_rel_change_ntk']:7.3f} | {r['rel_dist_net_vs_krr_ntk_init']:6.3f} {r['rel_dist_net_vs_krr_asign_init']:6.3f} "
          f"{r['rel_dist_net_vs_krr_sign_init']:6.3f} {r['rel_dist_net_vs_krr_ntk_inf']:6.3f} | {r['agree_net_vs_krr_ntk_init']:5.3f} "
          f"{r['agree_net_vs_krr_asign_init']:6.3f} {r['agree_net_vs_krr_sign_init']:5.3f}")
