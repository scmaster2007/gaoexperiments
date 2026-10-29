"""H4 analysis: does Adam end flatter, is its feedback acting on raw or preconditioned sharpness, and does
sharpness predict test performance?

    python3 experiments/h4_adam_sharpness/analyze_sharpness.py
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
MATCHED_LOSS = 0.1        # compare optimizers at the first checkpoint where the training loss drops below this
runs = json.load(open(os.path.join(HERE, "results.json")))
assert all(r["diverged_at"] is None for r in runs), "some runs diverged"


def at_matched_loss(history):
    return next((h for h in history if h["train_loss"] <= MATCHED_LOSS), None)


def edge_ratio(r, h):
    """Sharpness divided by the optimizer's stability threshold (raw H for GD/momentum, P^-1 H for Adam)."""
    return h.get("precond_sharpness", h["sharpness"]) / r["threshold"]


rows = []
for r in runs:
    hist, final, matched = r["history"], r["history"][-1], at_matched_loss(r["history"])
    late = [edge_ratio(r, h) for h in hist if h["step"] >= 1000]
    rows.append(dict(
        opt=r["opt"], lr=r["lr"], seed=r["seed"],
        sharp_matched=matched["sharpness"], trace_matched=matched["trace"], acc_matched=matched["test_acc"],
        step_matched=matched["step"],
        sharp_final=final["sharpness"], trace_final=final["trace"], acc_final=final["test_acc"],
        test_loss_final=final["test_loss"], train_loss_final=final["train_loss"],
        edge_ratio_final=edge_ratio(r, final),
        frac_at_edge=np.mean([0.85 <= e <= 1.15 for e in late])))
runs_df = pd.DataFrame(rows).sort_values(["opt", "lr", "seed"])
runs_df.to_csv(os.path.join(HERE, "per_run.csv"), index=False)

order = ["gd", "momentum", "adam", "adam_frozen"]
summary = runs_df.drop(columns="seed").groupby(["opt", "lr"]).median()
summary = summary.loc[[k for o in order for k in summary.index if k[0] == o]]
summary.to_csv(os.path.join(HERE, "summary.csv"))

print(f"1. At matched training loss ({MATCHED_LOSS}): median over 3 seeds")
print(summary[["step_matched", "sharp_matched", "trace_matched", "acc_matched"]].round(3).to_string(), "\n")
print("2. End of training (5000 steps)")
print(summary[["train_loss_final", "sharp_final", "trace_final", "acc_final", "test_loss_final"]].round(4).to_string(), "\n")
print("3. Feedback: sharpness / stability threshold at the end, and fraction of checkpoints after step 1000")
print("   within 15% of the threshold (raw H for gd/momentum, preconditioned P^-1 H for adam)")
print(summary[["edge_ratio_final", "frac_at_edge"]].round(3).to_string(), "\n")

print("4. Spearman across all 36 runs")
for x in ["sharp_matched", "trace_matched"]:
    rho, p = spearmanr(runs_df[x], runs_df["acc_matched"])
    print(f"   {x:14s} vs test acc at matched loss: rho = {rho:+.2f} (p = {p:.3f})")
for x in ["sharp_final", "trace_final"]:
    rho, p = spearmanr(runs_df[x], runs_df["acc_final"])
    print(f"   {x:14s} vs final test acc:          rho = {rho:+.2f} (p = {p:.3f})")
