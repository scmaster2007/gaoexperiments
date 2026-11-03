"""Checks kern.py's kernels against brute-force per-example gradients (CPU, tiny sizes)."""
import os
import runpy
import sys
import tempfile

import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
out = tempfile.mkdtemp()
data = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data")
sys.argv = ["kern.py", "--small", "--device", "cpu", "--depth", "3", "--width", "24", "--epochs", "2",
            "--max_epochs_run", "1", "--batch", "256", "--out", out, "--data", data, "--chunk_floats", "2000"]
G = runpy.run_path(os.path.join(here, "kern.py"), run_name="__main__")
model, opt, params, kernels, X_ntk = G["model"], G["opt"], G["params"], G["kernels"], G["X_ntk"]
Xs = X_ntk[:12].clone()
ks, frac, dead = kernels(Xs)

# brute force: per-example Jacobians of each output wrt all parameters
names = [n for n, _ in model.named_parameters()]
pd = {n: q.detach() for n, q in model.named_parameters()}


def f_single(pp, x):
    return torch.func.functional_call(model, pp, (x[None],))[0]


J = torch.func.vmap(torch.func.jacrev(f_single), in_dims=(None, 0))(pd, Xs)  # dict name -> [n, 10, *shape]
Jf = torch.cat([J[n].reshape(Xs.shape[0], 10, -1) for n in names], dim=2).double()  # [n, 10, P]
st = [opt.state[q] for q in params]
t = int(st[0]["step"]); b2c = 1 - 0.999 ** t
sv = torch.cat([((s["exp_avg_sq"] / b2c).sqrt()).reshape(-1) for s in st]).double()
q = 1.0 / (sv + 1e-8)
live = (sv >= 1e-8).double()
Kstd = torch.einsum("ikp,jkp->ij", Jf, Jf) / 10
Kad = torch.einsum("ikp,p,jkp->ij", Jf, q, Jf) / 10
Kli = torch.einsum("ikp,p,jkp->ij", Jf, q * live, Jf) / 10
ok = True
for nm, a, b in (("K_std", ks["K_std"], Kstd), ("K_adam", ks["K_adam"], Kad), ("K_live", ks["K_live"], Kli)):
    b = b.numpy()
    rel = np.abs(a - b).max() / np.abs(b).max()
    print(f"{nm}: max rel err {rel:.2e}  (scale {np.abs(b).max():.3e})")
    ok &= rel < 1e-5
print("frac sqrt(vhat) < eps:", frac, " dead:", dead)
meta = G["meta"]
print("checkpoints saved:", [c["epoch"] for c in meta["ckpts"]], "kernels:", meta["ckpts"][-1].get("kernels"),
      "hessian:", meta["hessian"], "errors:", meta["errors"])
print("PASS" if ok else "FAIL")
