"""Writes jobs/main_grid.txt: the main grid, one rich.py command per line, highest priority first."""
J = []
def job(opt="adam", L=4, w=1024, b=128, lr=1e-3, eps=1e-8, sched="const", ep=500, s=0, extra=""):
    J.append(f"python3 -u rich.py --opt {opt} --depth {L} --width {w} "
             f"--batch {b} --lr {lr} --eps {eps} --sched {sched} --epochs {ep} --seed {s} {extra}".strip())
D = (2, 4, 8, 16)
for s in range(5):                      # A: Adam defaults, 5 seeds
    for L in D: job(L=L, s=s)
for s in range(3):                      # B: SGD momentum 0.9, two learning rates
    for lr in (0.01, 0.05):
        for L in D: job(opt="sgd", L=L, lr=lr, s=s)
for s in range(3):                      # C: Adam with cosine decay over 150 and 500 epochs
    for ep in (150, 500):
        for L in D: job(L=L, sched="cosine", ep=ep, s=s)
for s in range(3):                      # D: Adam eps variants
    for eps in (1e-12, 1e-4):
        for L in D: job(L=L, eps=eps, s=s)
for s in range(3):                      # E: AdamW (PyTorch default weight decay 0.01)
    for L in D: job(opt="adamw", L=L, s=s)
for s in range(3):                      # D2: Adam smaller learning rates
    for lr in (3e-4, 1e-4):
        for L in D: job(L=L, lr=lr, s=s)
for s in range(3):                      # F: batch-size sweep
    for b in (32, 64, 256, 512):
        for L in (4, 16): job(L=L, b=b, s=s)
for s in range(3):                      # G: width
    for w in (256, 2048):
        for L in (4, 16): job(L=L, w=w, s=s)
for w in (256, 2048):
    for L in (4, 16): job(opt="sgd", L=L, w=w, lr=0.01, s=0)
open("jobs/main_grid.txt", "w").write("\n".join(J) + "\n")
print(len(J), "jobs")
