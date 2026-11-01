"""Shared pieces: CIFAR-10 loading, MLPs, exact empirical NTK for MLPs, the infinite-width
ReLU NTK/NNGP, and linearized (kernel) predictions.

Conventions
- Inputs: CIFAR-10 pixels standardized per channel with training-set statistics, flattened to 3072.
- Labels: one-hot, 10 outputs. Loss = 0.5 * mean over examples of the squared error summed over outputs.
- MLP with L hidden ReLU layers and a linear readout. Bias-free unless stated.
  'ntk' parametrization: W ~ N(0, 1), forward multiplies by sqrt(sw2 / fan_in) (readout: sqrt(so2 / fan_in)).
  'standard' parametrization: W ~ N(0, sw2 / fan_in), no forward multiplier (readout variance so2 / fan_in).
- Kernels are averaged over the 10 outputs (the infinite-width NTK is the same for every output).
"""

import math
import os
import pickle
import tarfile

import numpy as np
import torch

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# ----------------------------------------------------------------------------- data

def load_cifar(root):
    """Returns float32 arrays x_train [50000, 3072], y_train, x_test [10000, 3072], y_test.
    Accepts either the official python tarball in `root` or a small local npz (for CPU tests)."""
    npz = os.path.join(root, "cifar_subset.npz")
    tgz = os.path.join(root, "cifar-10-python.tar.gz")
    if os.path.exists(tgz):
        xs, ys = [], []
        with tarfile.open(tgz) as tf:
            def get(name):
                d = pickle.load(tf.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
                return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])
            for i in range(1, 6):
                x, y = get(f"data_batch_{i}")
                xs.append(x); ys.append(y)
            xtr, ytr = np.concatenate(xs), np.concatenate(ys)
            xte, yte = get("test_batch")
    elif os.path.exists(npz):
        d = np.load(npz)
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    else:
        raise FileNotFoundError(f"no CIFAR-10 in {root}")
    xtr = xtr.astype(np.float32) / 255.0
    xte = xte.astype(np.float32) / 255.0
    mean = xtr.reshape(-1, 3).mean(0)
    std = xtr.reshape(-1, 3).std(0)
    xtr = ((xtr - mean) / std).reshape(len(xtr), -1)
    xte = ((xte - mean) / std).reshape(len(xte), -1)
    return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)


def balanced_subset(y, n, seed=0):
    """First n examples, n/10 per class, in a fixed random order (seed-independent of model seeds)."""
    rng = np.random.default_rng(seed)
    per = n // 10
    idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:per] for c in range(10)])
    return np.sort(idx)


def onehot(y, k=10):
    return np.eye(k, dtype=np.float32)[y]


# ----------------------------------------------------------------------------- models

class MLP(torch.nn.Module):
    """Bias-free ReLU MLP with explicit parametrization; forward can return pre-activations."""

    def __init__(self, d, width, depth, out=10, param="ntk", sw2=2.0, so2=1.0, seed=0, device="cpu",
                 dtype=torch.float32):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        dims = [d] + [width] * depth
        self.param, self.depth = param, depth
        self.Ws = torch.nn.ParameterList()
        self.scales = []
        for l in range(depth):
            fan_in = dims[l]
            W = torch.randn(dims[l + 1], fan_in, generator=g, dtype=dtype)
            if param == "ntk":
                self.scales.append(math.sqrt(sw2 / fan_in))
            else:
                W *= math.sqrt(sw2 / fan_in)
                self.scales.append(1.0)
            self.Ws.append(torch.nn.Parameter(W.to(device)))
        W = torch.randn(out, width, generator=g, dtype=dtype)
        if param == "ntk":
            self.s_out = math.sqrt(so2 / width)
        else:
            W *= math.sqrt(so2 / width)
            self.s_out = 1.0
        self.Wout = torch.nn.Parameter(W.to(device))

    def forward(self, x, keep=False):
        h = x
        zs, hs = [], [x]
        for W, s in zip(self.Ws, self.scales):
            z = (h @ W.T) * s
            if keep:
                zs.append(z)
            h = torch.relu(z)
            if keep:
                hs.append(h)
        f = (h @ self.Wout.T) * self.s_out
        if keep:
            return f, zs, hs
        return f


def realistic_mlp(d, width, depth, out=10, seed=0, device="cpu"):
    """PyTorch-default MLP (nn.Linear default init, with biases): the 'realistic' setting."""
    torch.manual_seed(seed)
    layers, k = [], d
    for _ in range(depth):
        layers += [torch.nn.Linear(k, width), torch.nn.ReLU()]
        k = width
    layers.append(torch.nn.Linear(k, out))
    return torch.nn.Sequential(*layers).to(device)


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


# ----------------------------------------------------------------------------- empirical NTK

def empirical_ntk(model, X, chunk=4096):
    """Exact output-averaged empirical NTK of a bias-free MLP on inputs X ([n, d]).
    Uses the per-layer identity <dF/dW_l(x), dF/dW_l(x')> = s_l^2 <delta_l(x), delta_l(x')> <h_{l-1}(x), h_{l-1}(x')>,
    so no per-example gradients are formed."""
    n = X.shape[0]
    K = torch.zeros(n, n, dtype=torch.float64, device=X.device)
    Xg = X.detach()
    f, zs, hs = model(Xg, keep=True)
    # readout contribution: identical for every output
    hL = hs[-1].detach().double()
    K += (model.s_out ** 2) * (hL @ hL.T)
    Hgram = [(h.detach().double() @ h.detach().double().T) for h in hs[:-1]]  # input to each hidden layer's weight
    k_out = f.shape[1]
    for k in range(k_out):
        grads = torch.autograd.grad(f[:, k].sum(), zs, retain_graph=(k < k_out - 1))
        for l, dl in enumerate(grads):
            dl = dl.detach().double()
            K +=(model.scales[l] ** 2) * (dl @ dl.T) * Hgram[l] / k_out
    return K


# ----------------------------------------------------------------------------- infinite-width kernels

def relu_kernels(X1, X2, depth, sw2=2.0, so2=1.0, save_depths=None, dtype=torch.float64):
    """Infinite-width NNGP and NTK of a bias-free ReLU MLP with `depth` hidden layers (NTK parametrization),
    between rows of X1 and X2. Returns (nngp, ntk) at the readout; with save_depths, a dict depth -> (nngp, ntk)
    for networks of each listed depth (computed in one pass)."""
    d = X1.shape[1]
    X1 = X1.to(dtype)
    X2 = X2.to(dtype)
    S = sw2 * (X1 @ X2.T) / d
    q1 = sw2 * (X1 * X1).sum(1) / d
    q2 = sw2 * (X2 * X2).sum(1) / d
    T = S.clone()
    out = {}
    for l in range(1, depth + 1):
        # S, T are the layer-l pre-activation covariance and NTK; apply ReLU then a weight layer.
        norm = torch.sqrt(q1[:, None] * q2[None, :])
        c = torch.clamp(S / norm, -1.0, 1.0)
        th = torch.arccos(c)
        Ephi = norm * (torch.sin(th) + (math.pi - th) * c) / (2 * math.pi)
        Edphi = (math.pi - th) / (2 * math.pi)
        if save_depths is not None and l in save_depths:
            # a network with exactly l hidden layers would apply the readout here
            out[l] = (so2 * Ephi, so2 * Ephi + so2 * Edphi * T)
        if l == depth:
            return (so2 * Ephi, so2 * Ephi + so2 * Edphi * T) if save_depths is None else out
        S = sw2 * Ephi
        T = S + sw2 * Edphi * T
        q1 = sw2 * q1 / 2
        q2 = sw2 * q2 / 2
    return out


# ----------------------------------------------------------------------------- linearized predictions

def linear_gd_predict(Ktt, Kst, f0_t, f0_s, Y, eta, steps, n):
    """Outputs after `steps` of full-batch GD on the model linearized with kernel K (output-averaged, same for
    every output), loss 0.5 * mean ||f - y||^2. steps=None gives the t -> infinity limit (kernel regression).
    All tensors float64."""
    lam, V = torch.linalg.eigh(Ktt)
    lam = torch.clamp(lam, min=0.0)
    r0 = f0_t - Y
    c = V.T @ r0
    if steps is None:
        decay = torch.zeros_like(lam)
    else:
        decay = (1.0 - eta * lam / n) ** steps
    f_train = Y + V @ (decay[:, None] * c)
    # test: f_s = f0_s - Kst V diag((1 - decay) / lam) V^T r0
    w = torch.where(lam > lam.max() * 1e-12, (1.0 - decay) / torch.clamp(lam, min=1e-300), torch.zeros_like(lam))
    f_test = f0_s - Kst @ (V @ (w[:, None] * c))
    return f_train, f_test


def kernel_regression(Ktt, Kst, Y, ridge_rel=1e-6):
    """Kernel regression f(x) = K(x, X) (K + ridge I)^{-1} Y with ridge = ridge_rel * mean diagonal (float64)."""
    n = Ktt.shape[0]
    r = ridge_rel * torch.diagonal(Ktt).mean()
    A = Ktt + r * torch.eye(n, dtype=Ktt.dtype, device=Ktt.device)
    L = torch.linalg.cholesky(A)
    alpha = torch.cholesky_solve(Y, L)
    return Kst @ alpha
