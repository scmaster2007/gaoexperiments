"""Rebuild law_data.npz on the node from the enumeration output and the planted list."""
import numpy as np, pandas as pd
d = pd.read_csv("data/behaviours.csv")
w = 5.0 ** -np.arange(14)
p_nat = (d[[f"n{L}" for L in range(14)]].values * w).sum(1); p_nat /= p_nat.sum()
A = pd.read_csv("runs/exp4_scale/plant_min.csv")
assert (d.tt.values[A.idx.values] == A.tt.values).all()
p = p_nat.copy()
keep = np.ones(len(p), bool); keep[A.idx.values] = False
p[A.idx.values] = A.freq.values
p[keep] = p_nat[keep] / p_nat[keep].sum() * (1 - A.freq.sum())
np.savez("runs/exp4_scale/law_data.npz", tt=d.tt.values.astype(np.int64), p=p, natural_p=p_nat,
         plant_idx=A.idx.values, plant_level=A.level.values, plant_freq=A.freq.values)
print("checksum tt:", int(d.tt.values.sum()), " p·index:", repr(float((p * np.arange(len(p))).sum())), " n:", len(p))
