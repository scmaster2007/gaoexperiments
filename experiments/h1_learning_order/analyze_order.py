"""H1 analysis: do networks share a learning order over behaviours, and which complexity measure predicts it?

Accessibility of a behaviour = its accuracy averaged over all training checkpoints (as in "Let's agree to
agree"). We compare it across seeds and widths, and against complexity measures, before and after
controlling for (a) how often the behaviour appears in training, log2 P(b), and (b) how predictable it is
for the exact Bayes learner (its Bayes-optimal accuracy on the same input orders).

    python3 experiments/h1_learning_order/analyze_order.py
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from shared import FEATURES_PATH, PRIOR_PATH, bayes_predict, eval_set, load_prior

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURES = ["K", "log2p", "avg_sensitivity", "lz", "log2_runtime", "relevant_vars", "fourier_degree"]


def bayes_accuracy(prior, orders=4):
    """Per-behaviour accuracy of the exact Bayes predictor on the same input orders the networks saw."""
    xs, ys = eval_set(prior["tt"].values.astype(np.int64), orders)
    correct = []
    for i in range(0, len(xs), 1024):
        p1 = bayes_predict(prior, xs[i:i + 1024], ys[i:i + 1024])
        y = ys[i:i + 1024]
        correct.append(np.where(p1 == 0.5, 0.5, (p1 > 0.5) == y))       # ties count as a coin flip
    return np.concatenate(correct).mean(1).reshape(len(prior), orders).mean(1)


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after regressing the ranks of both on the ranks of the controls."""
    C = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls])
    resid = lambda v: rankdata(v) - C @ np.linalg.lstsq(C, rankdata(v), rcond=None)[0]
    return np.corrcoef(resid(x), resid(y))[0, 1]


prior = load_prior(PRIOR_PATH)
features = pd.read_csv(FEATURES_PATH).set_index("tt").loc[prior["tt"]].reset_index()
features["bayes_acc"] = bayes_accuracy(prior)

runs = {os.path.basename(p)[:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(HERE, "runs/*.npz")))}
access = pd.DataFrame({name: r["acc"].mean(0) for name, r in runs.items()})
final = pd.DataFrame({name: r["acc"][-1] for name, r in runs.items()})
names = list(access.columns)

# 1. agreement of the learning order across seeds and widths
print("1. Spearman correlation of accessibility between runs")
print(access.corr("spearman").round(3).to_string(), "\n")
residual = access.sub(features["bayes_acc"], axis=0)     # what the network finds easy beyond what Bayes finds easy
print("   same, after subtracting each behaviour's Bayes-optimal accuracy")
print(residual.corr("spearman").round(3).to_string(), "\n")

# 2. which complexity measure predicts accessibility
mean_access = access.mean(1)
rows = []
for m in MEASURES + ["bayes_acc"]:
    rows.append({"measure": m,
                 "rho": spearmanr(features[m], mean_access)[0],
                 "rho | log2p": partial_spearman(features[m], mean_access, [features["log2p"]]) if m != "log2p" else np.nan,
                 "rho | log2p, bayes_acc": partial_spearman(features[m], mean_access, [features["log2p"], features["bayes_acc"]])
                 if m not in ("log2p", "bayes_acc") else np.nan})
measures = pd.DataFrame(rows).set_index("measure").round(3)
print("2. Spearman of mean accessibility (all 6 runs) with each measure, raw and partial")
print(measures.to_string(), "\n")

# 3. does the weaker model learn a subset of what the stronger model learns?
learned = final.ge(features["bayes_acc"] - 0.02, axis=0)        # within 2 points of Bayes-optimal at the end
small = learned[[n for n in names if n.startswith("d32")]].mean(1) > 0.5
large = learned[[n for n in names if n.startswith("d64")]].mean(1) > 0.5
print("3. Behaviours learned (final accuracy within 0.02 of Bayes, majority of seeds)")
print(f"   d32: {small.sum()}  d64: {large.sum()}  both: {(small & large).sum()}")
print(f"   P(learned by d64 | learned by d32) = {(small & large).sum() / small.sum():.3f}"
      f"   base rate P(learned by d64) = {large.mean():.3f}\n")

# 4. accessibility by shortest program length
by_k = pd.DataFrame({"n": features.groupby("K").size(),
                     "accessibility": mean_access.groupby(features["K"]).mean(),
                     "bayes_acc": features.groupby("K")["bayes_acc"].mean(),
                     "access - bayes": residual.mean(1).groupby(features["K"]).mean()}).round(3)
print("4. By shortest program length K")
print(by_k.to_string())

access.assign(tt=prior["tt"], bayes_acc=features["bayes_acc"]).to_csv(os.path.join(HERE, "accessibility.csv"), index=False)
measures.to_csv(os.path.join(HERE, "measures.csv"))
by_k.to_csv(os.path.join(HERE, "by_K.csv"))
