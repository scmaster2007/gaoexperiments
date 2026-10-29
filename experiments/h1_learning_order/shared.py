"""Pieces shared by the H1 training and analysis scripts."""
import os, sys
import numpy as np

AIXI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../aixi")
sys.path.insert(0, os.path.join(AIXI, "src"))
from common import load_prior, bayes_predict, NIN  # noqa: E402

PRIOR_PATH = os.path.join(AIXI, "data/behaviours.csv")
FEATURES_PATH = os.path.join(AIXI, "results/results_per_behaviour.csv")


def eval_set(tt, orders=4, seed=777):
    """Every behaviour shown in `orders` fixed random input orders. Row i*orders + r is behaviour i, order r."""
    rng = np.random.default_rng(seed)
    xs = np.argsort(rng.random((len(tt) * orders, NIN)), axis=1)
    ys = (np.repeat(tt, orders)[:, None] >> xs) & 1
    return xs, ys
