"""H3 analysis: learning order on the staircase, and how learning time scales with input dimension d.

For each (target, optimizer, lr) we fit log T = a + b log d over seeds, where T is the step at which the
highest-degree monomial is learned. SGD theory predicts b ~ 1 for leap1 and b ~ 2 for leap3.

    python3 experiments/h3_leap_adam/analyze_leap.py
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
runs = pd.DataFrame(json.load(open(os.path.join(HERE, "results.json"))))
runs["T"] = runs.t_learn.str[-1]                                       # step the last monomial is learned
assert runs["T"].notna().all(), "some runs never learned the target"

# 1. learning order on the staircase: are x1, x1x2, x1x2x3 learned in that order?
stair = runs[runs.target == "leap1"].copy()
stair[["t1", "t2", "t3"]] = pd.DataFrame(stair.t_learn.tolist(), index=stair.index)
stair["in_order"] = (stair.t1 <= stair.t2) & (stair.t2 <= stair.t3)
order = stair.groupby(["opt", "lr", "d"])[["t1", "t2", "t3"]].median().astype(int)
order["frac_in_order"] = stair.groupby(["opt", "lr", "d"]).in_order.mean()
print("1. leap1 staircase: median step each monomial is learned\n" + order.to_string() + "\n")

# 2. scaling of learning time with d
fits = []
for (target, opt, lr), g in runs.groupby(["target", "opt", "lr"]):
    slope, intercept = np.polyfit(np.log(g.d), np.log(g["T"]), 1)
    resid = np.log(g["T"]) - (intercept + slope * np.log(g.d))
    se = np.sqrt(resid.var(ddof=2) / ((np.log(g.d) - np.log(g.d).mean()) ** 2).sum())
    medians = g.groupby("d")["T"].median().astype(int).to_dict()
    fits.append(dict(target=target, opt=opt, lr=lr, exponent=slope, se=se, **{f"T(d={d})": t for d, t in medians.items()}))
fits = pd.DataFrame(fits).round({"exponent": 2, "se": 2})
print("2. T ~ d^exponent (median T per d shown)\n" + fits.to_string(index=False))
fits.to_csv(os.path.join(HERE, "scaling.csv"), index=False)
order.to_csv(os.path.join(HERE, "staircase_order.csv"))
