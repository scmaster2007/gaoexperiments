"""Writes jobs/width_and_data_order.txt: the Adam-at-width sweep (NTK parametrization) and the
init-seed vs data-order-seed runs."""
J = []
def P(L, w): return 3072 * w + (L - 1) * w * w
aw = []
for L in (2, 4, 8):
    for w in (256, 1024, 4096, 8192):
        for p in (0.0, 0.5, 1.0):
            for s in (0, 1):
                aw.append((P(L, w), f"python3 -u adamwide.py "
                                    f"--depth {L} --width {w} --p {p} --seed {s}"))
aw.sort(key=lambda x: -x[0])
J += [a[1] for a in aw]
for L in (4, 16):
    for opt, lr in (("adam", 0.001), ("sgd", 0.01)):
        for s, ds in ((0, 1), (0, 2), (1, 0), (2, 0)):
            J.append(f"python3 -u rich.py --opt {opt} --depth {L} --width 1024 "
                     f"--batch 128 --lr {lr} --eps 1e-08 --sched const --epochs 500 --seed {s} --data_seed {ds}")
open("jobs/width_and_data_order.txt", "w").write("\n".join(J) + "\n")
print(len(J), "extra jobs")
