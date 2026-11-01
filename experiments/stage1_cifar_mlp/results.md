# Stage 1 data: Adam on deep MLPs, CIFAR-10

Oct 31, 2026

## What was run

There are 173 of 184 planned 500-epoch training runs, 71 of 72 Adam-at-width runs, and four sets of known-result checks.

| Set | Runs | Setup |
| --- | --- | --- |
| Main grid | 173 of 184 | CIFAR-10, all 50,000 training and 10,000 test images; flattened, standardized pixels. ReLU MLP in PyTorch's default parametrization (nn.Linear init, with biases), width 1024 unless noted; squared loss on one-hot labels; 500 epochs. Adam defaults: lr 10⁻³, β (0.9, 0.999), ε 10⁻⁸, batch 128, constant lr. |
| Variations in the grid | — | Depths 2, 4, 8, 16. SGD with momentum 0.9 at lr 0.01 and 0.05. Cosine lr decay over 150 or 500 epochs. ε = 10⁻¹² and 10⁻⁴. AdamW (weight decay 0.01). lr 3×10⁻⁴ and 10⁻⁴. Batch sizes 32–512. Widths 256 and 2048. Runs with init seed ≠ data-order seed. |
| Adam at width (NTK parametrization) | 71 of 72 | See that section. |
| Known results (R1–R4) | 148 width-sweep runs, 105 trainability runs, kernel computations | See that section. |

**What each run recorded:**

- **Every step:** training loss.
- **About 70 checkpoints:** outputs on all 10,000 test images, outputs on a fixed 2,000-image training subset, and which training images are classified correctly.
- **13 checkpoints (epochs 0, 1, 3, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500):** weight norms and singular values; distance from init; per-neuron incoming and outgoing weight norms; per-layer inactive fraction and effective rank of the representation; NTK on 500 probe images; top Hessian eigenvalue; for Adam, the preconditioned top eigenvalue and second-moment statistics.
- **Weights:** final weights, plus epoch-150 weights for seed 0.

**Missing from the grid (11):**

- Batch 32 at depths 4 (seeds 0, 1) and 16 (seeds 0–2): each needs 4× the steps of a batch-128 run.
- Batch 64 at depth 16 (seeds 0–2): stopped at epoch 50; a stopped run saves nothing.
- Batch 512 at depth 16 (seed 2) and depth 4 (seeds 1, 2): skipped at the end.

The credit ran out once (about 15:00–21:00 UTC Oct 31) and all nodes stopped. No data was lost.

## Raw data

All raw data is in 15 archives, stored as givemeanode artifacts until about Nov 30, 2026; `get_artifact(<id>)` returns a download link. Each directory archive is a tar file without the .tar extension: download it, then run `tar -xf <file>`.

| Archive | Runs | Size | Artifact |
| --- | --- | --- | --- |
| Summaries: rich\_summary.json, post.json, compact.json, adamwide\_summary.json, R1–R4 figures and summary.json, text tables | — | 2.9 MB | `art-picmv` |
| Main grid, node h1 | 42 | 2.39 GB | `art-uvf8c` |
| Main grid, node h2 (to 14:17 UTC) | 38 | 1.93 GB | `art-gzwij` |
| Main grid, node h2 (evening runs) | 15 | 1.08 GB | `art-p3ui4` |
| Main grid, node h3 | 31 | 1.45 GB | `art-sez7u` |
| Main grid, node h4 | 24 | 1.08 GB | `art-7subr` |
| Main grid, node h5 | 23 | 1.05 GB | `art-99yyf` |
| Adam at width, node h1 | 14 | 3.08 GB | `art-v7q9a` |
| Adam at width, node h2 | 15 | 3.29 GB | `art-hrtan` |
| Adam at width, node h3 | 14 | 3.07 GB | `art-axk3w` |
| Adam at width, node h4 | 14 | 3.08 GB | `art-xe6dv` |
| Adam at width, node h5 | 14 | 3.07 GB | `art-38tzi` |
| Known results R1–R4 and pilots, all nodes | 409 files | 0.84 GB | `art-zpr2x` |
| Code and logs, node h1 | — | 0.6 MB | `art-v6y94` |
| Code and logs, node h2 | — | 0.4 MB | `art-uz2ib` |

**File formats.**

- **Main grid:** each run is rich\_\<config>\_s\<seed>\[\_ds\<data seed>\].json plus a .npz.
  - The JSON holds the config and per-checkpoint curves: step, epoch, train/test loss and accuracy.
  - The .npz holds test\_out (checkpoints × 10,000 × 10, fp16); train\_out\_probe (on probe\_train\_out); train\_correct\_bits (packed, all 50,000 training images); the deep\_\* arrays at the 13 deep checkpoints; and weights\_step\<s>\_w\<l> / \_b\<l>.
- **Adam at width:** aw\_L\<depth>\_w\<width>*p\<p>*s\<seed>.npz holds the output trajectory on 3,000 points (2,000 train then 1,000 test), every kernel predictor, and the six 3,000 × 3,000 kernels at init and end (K0*\*, KT*\*) plus Kinf\_ntk.
- **R1–R4:** r12\_*, r3sig\_*, r3train\_*, r4\_* and pilot\_\* files.
- **Code:** the code archives contain every script used, including rich.py, adamwide.py, repro.py, core.py, sum\_rich.py and post.py.

## Known results reproduced (R1–R4)

The four checks run in the NTK parametrization (weight variance 2/fan-in, no biases, squared loss). Theory's reference value is given beside each measured one.

**R1 and R2: kernel and lazy-training limits.** 2,000 CIFAR-10 training and 1,000 test images. Full-batch gradient descent for 2,000 steps; learning rate 1/(λ\_max/n) of the infinite-width NTK. 6 seeds per width, except at the largest widths (1–4 seeds). Each cell gives the value at width 256 → the largest width, and in brackets the log-log slope against width.

| Hidden layers | Widths | Init NTK error vs infinite width | Weight change ‖θ\_T−θ\_0‖/‖θ\_0‖ | Kernel change ‖K\_T−K\_0‖/‖K\_0‖ | Test outputs vs own linearization | Test outputs vs infinite-width linear model |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 256–16384 | 0.203 → 0.038 (−0.42) | 0.0286 → 0.0019 (−0.66) | 0.728 → 0.034 (−0.75) | 0.577 → 0.138 (−0.35) | 0.689 → 0.155 (−0.36) |
| 4 | 256–16384 | 0.286 → 0.044 (−0.40) | 0.0175 → 0.0009 (−0.73) | 0.770 → 0.059 (−0.62) | 0.652 → 0.192 (−0.30) | 0.741 → 0.209 (−0.31) |
| 8 | 256–16384 | 0.360 → 0.037 (−0.55) | 0.0094 → 0.0004 (−0.77) | 0.813 → 0.096 (−0.52) | 0.756 → 0.269 (−0.26) | 0.818 → 0.289 (−0.25) |
| 16 | 256–16384 | 0.377 → 0.075 (−0.42) | 0.0046 → 0.0002 (−0.80) | 0.809 → 0.081 (−0.54) | 0.836 → 0.353 (−0.21) | 0.884 → 0.366 (−0.21) |
| 32 | 256–8192 | 0.555 → 0.062 (−0.58) | 0.0021 → 0.0001 (−0.79) | 0.749 → 0.050 (−0.91) | 0.837 → 0.556 (−0.13) | 0.899 → 0.542 (−0.16) |

Theory's rate for the initial kernel error is width^−1/2 (slope −0.5). The seed-averaged network output was also compared with the infinite-width mean predictor. That distance went from 0.53 at width 256 to 1.85 at width 16384 (2 hidden layers); the 16384 point has 1 seed.

**R3a: signal propagation at initialization** (100 hidden layers, width 4096). Pre-activation variance at layer 100 divided by that at layer 1, by weight variance × fan-in:

| σ\_w² | 1.0 | 1.5 | 1.8 | 1.9 | 2.0 | 2.1 | 2.2 | 2.5 | 3.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q₁₀₀ / q₁ | 1.7e−30 | 4.5e−13 | 3.1e−5 | 6.6e−3 | 1.06 | 133 | 1.3e4 | 4.2e9 | 2.9e17 |

Measured over theoretical variance at layers 1, 10 and 100 is between 0.97 and 1.08 for every σ\_w². At σ\_w² = 2, the correlation between two inputs at layers 1, 10 and 100 is 0.166, 0.865 and 0.9965; theory gives 0.165, 0.867 and 0.9964.

**R3b: trainability with SGD.** Training accuracy, best learning rate per cell; 0 means the run diverged.

| Hidden layers | σ\_w² 1.0 | 1.5 | 1.8 | 2.0 | 2.2 | 2.5 | 3.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.68 | 0.69 | 0.63 | 0.59 | 0.43 | 0.42 | 0 |
| 16 | 0.65 | 0.65 | 0.63 | 0.58 | 0.41 | 0 | 0 |
| 32 | 0.10 | 0.57 | 0.59 | 0.50 | 0 | 0 | 0 |
| 64 | 0.10 | 0.16 | 0.51 | 0.48 | 0 | 0 | 0 |
| 128 | 0.10 | 0.10 | 0.24 | 0.31 | 0 | 0 | 0 |

**R4: the infinite-width NTK across depth** (σ\_w² = 2; kernel regression on 20,000 training images, 10,000 test images; best of 4 ridges).

| Hidden layers | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NTK regression test accuracy | 0.521 | 0.536 | 0.530 | 0.518 | 0.495 | 0.475 | 0.448 | 0.420 |
| NNGP regression test accuracy | 0.509 | 0.535 | 0.539 | 0.538 | 0.530 | 0.520 | 0.499 | 0.483 |
| NTK off-diagonal / diagonal mean | 0.319 | 0.340 | 0.330 | 0.307 | 0.281 | 0.260 | 0.246 | 0.238 |
| Spread of NTK predictions across test inputs | 0.201 | 0.177 | 0.163 | 0.148 | 0.134 | 0.120 | 0.106 | 0.093 |

- NTK diagonal against depth, depths 16 and up: log-log slope 0.98 (theory 1, linear growth).
- Spread of predictions against depth, depths 16 and up: slope −0.17 at n = 20,000 and −0.25 at n = 5,000 (depths to 512). The theory's 1/depth would be −1.
- NTK condition number on m random Gaussian inputs at depth 512: 5.13 for m = 12 and 7.88 for m = 20; theory's limit (m+3)/3 gives 5.0 and 7.67. On m CIFAR images: 16.8 and 33.1.
- Full CIFAR-10 (50,000 training images), NTK regression: 0.577 at 1 hidden layer, 0.587 at 2.

**Pilot** (Adam defaults, width 1024, batch 128, 500 epochs, standard parametrization). Final training / test accuracy: 2 hidden layers 0.959 / 0.474; 4: 0.916 / 0.506; 8: 0.101 / 0.101; 16: 0.100 / 0.100.

## Adam in the NTK parametrization across width

71 runs: 2, 4 and 8 hidden layers; widths 256 to 8192; learning rate 10⁻³ × (1024/width)^p for p = 0, 0.5 and 1; 2 seeds each.

- **Setup:** bias-free ReLU MLP, NTK parametrization. Full-batch Adam (β₁ 0.9, β₂ 0.999, ε 10⁻⁸) for 2,000 steps. 2,000 balanced training and 1,000 test images; squared loss.
- **Kernels:** three, computed exactly on all 3,000 points at initialization and at the end, averaged over the 10 outputs. NTK ⟨∇f, ∇f′⟩; A-SignGD ⟨∇f, sign ∇f′⟩; SignGD ⟨sign ∇f, sign ∇f′⟩ (Malladi et al. 2023).
- **Kernel predictors:** kernel regression to convergence with each initial kernel, offset by the network's own initial outputs. Also with the infinite-width NTK, with that offset (NTK∞) and without it (NTK∞ mean).
- **Distance:** ‖f\_net − f\_pred‖ / ‖f\_pred‖ on test outputs.

| Hidden layers | Width | p | lr | Final train loss | Net test acc | ΔW rel | ΔK rel (NTK / A-sign / sign) | Distance to NTK₀ | to A-SignGD₀ | to SignGD₀ | to NTK∞ | Class agreement with NTK∞ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 256 | 0 | 1.0e−3 | 0.0238 | 0.277 | 0.149 | 0.66 / 0.66 / 0.61 | 1.050 | 0.897 | 0.918 | 0.819 | 0.415 |
| 2 | 256 | 0.5 | 2.0e−3 | 0.0051 | 0.278 | 0.174 | 0.63 / 0.67 / 0.65 | 1.139 | 0.950 | 0.971 | 0.899 | 0.393 |
| 2 | 256 | 1 | 4.0e−3 | 0.0012 | 0.280 | 0.195 | 0.59 / 0.70 / 0.71 | 1.198 | 0.995 | 1.012 | 0.967 | 0.360 |
| 2 | 1024 | any | 1.0e−3 | 0.0000 | 0.276 | 0.066 | 0.31 / 0.29 / 0.24 | 0.724 | 0.675 | 0.761 | 0.589 | 0.571 |
| 2 | 4096 | 0 | 1.0e−3 | 0.0000 | 0.253 | 0.025 | 0.23 / 0.20 / 0.16 | 0.444 | 0.473 | 0.654 | 0.412 | 0.686 |
| 2 | 4096 | 0.5 | 5.0e−4 | 0.0005 | 0.247 | 0.024 | 0.11 / 0.09 / 0.07 | 0.435 | 0.463 | 0.648 | 0.397 | 0.698 |
| 2 | 4096 | 1 | 2.5e−4 | 0.0000 | 0.248 | 0.023 | 0.07 / 0.06 / 0.05 | 0.427 | 0.455 | 0.643 | 0.387 | 0.704 |
| 2 | 8192 | 0 | 1.0e−3 | 0.0078 | 0.283 | 0.015 | 0.28 / 0.25 / 0.21 | 0.383 | 0.417 | 0.598 | 0.375 | 0.708 |
| 2 | 8192 | 0.5 | 3.5e−4 | 0.0010 | 0.272 | 0.014 | 0.06 / 0.05 / 0.04 | 0.338 | 0.375 | 0.574 | 0.327 | 0.748 |
| 2 | 8192 | 1 | 1.25e−4 | 0.0000 | 0.269 | 0.013 | 0.03 / 0.02 / 0.02 | 0.328 | 0.366 | 0.570 | 0.317 | 0.755 |
| 4 | 256 | 0 | 1.0e−3 | 0.0009 | 0.268 | 0.130 | 0.68 / 0.63 / 0.62 | 1.128 | 0.970 | 0.959 | 0.900 | 0.356 |
| 4 | 256 | 0.5 | 2.0e−3 | 0.0000 | 0.256 | 0.142 | 0.71 / 0.67 / 0.66 | 1.164 | 1.006 | 1.004 | 0.954 | 0.329 |
| 4 | 256 | 1 | 4.0e−3 | 0.0000 | 0.274 | 0.156 | 0.75 / 0.73 / 0.72 | 1.168 | 1.030 | 1.030 | 0.988 | 0.300 |
| 4 | 1024 | any | 1.0e−3 | 0.0000 | 0.266 | 0.053 | 0.53 / 0.46 / 0.35 | 0.770 | 0.719 | 0.792 | 0.671 | 0.486 |
| 4 | 4096 | 0 | 1.0e−3 | 0.0000 | 0.287 | 0.018 | 0.60 / 0.50 / 0.35 | 0.522 | 0.534 | 0.653 | 0.518 | 0.609 |
| 4 | 4096 | 0.5 | 5.0e−4 | 0.0003 | 0.272 | 0.015 | 0.35 / 0.28 / 0.20 | 0.467 | 0.473 | 0.605 | 0.447 | 0.650 |
| 4 | 4096 | 1 | 2.5e−4 | 0.0000 | 0.266 | 0.014 | 0.18 / 0.15 / 0.11 | 0.452 | 0.452 | 0.589 | 0.421 | 0.676 |
| 4 | 8192 | 0 | 1.0e−3 | 0.0000 | 0.283 | 0.012 | 0.70 / 0.60 / 0.45 | 0.547 | 0.559 | 0.739 | 0.552 | 0.589 |
| 4 | 8192 | 0.5 | 3.5e−4 | 0.0002 | 0.258 | 0.008 | 0.27 / 0.21 / 0.15 | 0.384 | 0.395 | 0.651 | 0.378 | 0.709 |
| 4 | 8192 | 1 | 1.25e−4 | 0.0000 | 0.252 | 0.008 | 0.10 / 0.09 / 0.06 | 0.361 | 0.370 | 0.640 | 0.346 | 0.742 |
| 8 | 256 | 0 | 1.0e−3 | 0.0000 | 0.234 | 0.118 | 0.71 / 0.63 / 0.55 | 1.197 | 1.110 | 1.083 | 1.031 | 0.287 |
| 8 | 256 | 0.5 | 2.0e−3 | 0.0001 | 0.237 | 0.129 | 0.79 / 0.71 / 0.61 | 1.203 | 1.126 | 1.101 | 1.063 | 0.261 |
| 8 | 256 | 1 | 4.0e−3 | 0.0000 | 0.254 | 0.144 | 0.86 / 0.77 / 0.66 | 1.169 | 1.116 | 1.108 | 1.074 | 0.231 |
| 8 | 1024 | any | 1.0e−3 | 0.0000 | 0.255 | 0.041 | 0.68 / 0.55 / 0.36 | 0.876 | 0.856 | 0.890 | 0.810 | 0.390 |
| 8 | 4096 | 0 | 1.0e−3 | 0.0000 | 0.278 | 0.016 | 0.84 / 0.71 / 0.44 | 0.810 | 0.814 | 0.897 | 0.814 | 0.387 |
| 8 | 4096 | 0.5 | 5.0e−4 | 0.0000 | 0.255 | 0.011 | 0.60 / 0.47 / 0.27 | 0.633 | 0.636 | 0.761 | 0.617 | 0.520 |
| 8 | 4096 | 1 | 2.5e−4 | 0.0000 | 0.257 | 0.009 | 0.39 / 0.30 / 0.18 | 0.558 | 0.559 | 0.709 | 0.523 | 0.595 |
| 8 | 8192 | 0 | 1.0e−3 | 0.0000 | 0.359 | 0.016 | 0.97 / 0.91 / 0.67 | 0.840 | 0.846 | 0.941 | 0.855 | 0.348 |
| 8 | 8192 | 0.5 | 3.5e−4 | 0.0000 | 0.272 | 0.005 | 0.56 / 0.43 / 0.25 | 0.556 | 0.562 | 0.745 | 0.554 | 0.570 |
| 8 | 8192 | 1 | 1.25e−4 | 0.0000 | 0.244 | 0.004 | 0.27 / 0.20 / 0.11 | 0.437 | 0.443 | 0.685 | 0.421 | 0.663 |

Test accuracy of the kernel predictors, for comparison with the net column: NTK₀ 0.24–0.33, A-SignGD₀ 0.23–0.26, SignGD₀ 0.18–0.23, NTK∞ 0.23–0.28, NTK∞ mean 0.418–0.436. The row for 8 hidden layers, width 8192, p = 0 has 1 seed; the other seed ran out of GPU memory and was not rerun.

## Main grid: end-of-training metrics

At Adam's defaults, all 5 depth-8 seeds collapsed to chance between epochs 300 and 370. At depth 16, 3 of 5 seeds collapsed and 1 never reached 0.8 training accuracy. No depth-2 or depth-4 run met the collapse definition. No run at lr 10⁻⁴ or 3×10⁻⁴, or with cosine decay over 500 epochs, collapsed at any depth.

Width 1024 and batch 128 unless the variation says otherwise. Collapse epochs: “—” = reached 0.8 training accuracy and did not collapse; “n” = never reached 0.8.

| Optimizer | Hidden layers | Variation from defaults | Seeds | Final train acc | Final test acc | Peak test acc | Seed-avg test acc | Pairwise test agreement | Learning-time rank corr. | Collapse epoch per seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adam | 2 | defaults | 5 | 0.957 | 0.478 | 0.519 | 0.553 | 0.526 | 0.75 | — ×5 |
| Adam | 2 | lr 3e-4 | 3 | 0.999 | 0.550 | 0.555 | 0.586 | 0.629 | 0.79 | — ×3 |
| Adam | 2 | lr 1e-4 | 3 | 1.000 | 0.568 | 0.570 | 0.599 | 0.666 | 0.84 | — ×3 |
| Adam | 2 | ε 1e-12 | 3 | 0.959 | 0.474 | 0.516 | 0.533 | 0.525 | 0.75 | — ×3 |
| Adam | 2 | ε 1e-4 | 3 | 0.978 | 0.501 | 0.526 | 0.552 | 0.552 | 0.76 | — ×3 |
| Adam | 2 | cosine 150 | 3 | 0.957 | 0.473 | 0.517 | 0.531 | 0.505 | 0.76 | — ×3 |
| Adam | 2 | cosine 500 | 3 | 0.960 | 0.465 | 0.514 | 0.528 | 0.492 | 0.76 | — ×3 |
| AdamW | 2 | weight decay 0.01 | 3 | 0.965 | 0.490 | 0.521 | 0.544 | 0.521 | 0.74 | — ×3 |
| SGD | 2 | lr 0.01 | 3 | 1.000 | 0.534 | 0.558 | 0.584 | 0.586 | 0.82 | — ×3 |
| SGD | 2 | lr 0.05 | 3 | 0.999 | 0.536 | 0.551 | 0.581 | 0.591 | 0.78 | — ×3 |
| Adam | 4 | defaults | 5 | 0.932 | 0.505 | 0.531 | 0.574 | 0.538 | 0.61 | — ×5 |
| Adam | 4 | data order ≠ init | 4 | 0.962 | 0.510 | 0.531 | 0.565 | 0.545 | 0.61 | — ×4 |
| Adam | 4 | lr 3e-4 | 3 | 0.995 | 0.528 | 0.556 | 0.577 | 0.578 | 0.64 | — ×3 |
| Adam | 4 | lr 1e-4 | 3 | 0.999 | 0.559 | 0.572 | 0.597 | 0.632 | 0.68 | — ×3 |
| Adam | 4 | ε 1e-12 | 3 | 0.945 | 0.506 | 0.531 | 0.559 | 0.537 | 0.58 | — ×3 |
| Adam | 4 | ε 1e-4 | 3 | 0.997 | 0.530 | 0.543 | 0.577 | 0.581 | 0.63 | — ×3 |
| Adam | 4 | cosine 150 | 3 | 0.998 | 0.524 | 0.531 | 0.569 | 0.583 | 0.70 | — ×3 |
| Adam | 4 | cosine 500 | 3 | 0.995 | 0.503 | 0.530 | 0.559 | 0.528 | 0.64 | — ×3 |
| Adam | 4 | batch 32 | 1 | 0.242 | 0.193 | 0.527 | 0.193 |  |  | — |
| Adam | 4 | batch 64 | 3 | 0.579 | 0.355 | 0.526 | 0.506 | 0.305 | 0.23 | — ×3 |
| Adam | 4 | batch 256 | 3 | 0.994 | 0.518 | 0.532 | 0.570 | 0.557 | 0.62 | — ×3 |
| Adam | 4 | batch 512 | 1 | 0.997 | 0.521 | 0.536 | 0.521 |  |  | — |
| Adam | 4 | width 256 | 3 | 0.981 | 0.503 | 0.528 | 0.556 | 0.525 | 0.59 | — ×3 |
| Adam | 4 | width 2048 | 3 | 0.885 | 0.491 | 0.534 | 0.548 | 0.498 | 0.54 | — ×3 |
| AdamW | 4 | weight decay 0.01 | 3 | 0.986 | 0.524 | 0.534 | 0.566 | 0.571 | 0.56 | — ×3 |
| SGD | 4 | lr 0.01 | 3 | 1.000 | 0.567 | 0.573 | 0.595 | 0.668 | 0.81 | — ×3 |
| SGD | 4 | lr 0.01, data order ≠ init | 4 | 1.000 | 0.567 | 0.574 | 0.604 | 0.675 | 0.82 | — ×4 |
| SGD | 4 | lr 0.05 | 3 | 1.000 | 0.583 | 0.585 | 0.610 | 0.708 | 0.77 | — ×3 |
| SGD | 4 | lr 0.01, width 256 | 1 | 1.000 | 0.524 | 0.542 | 0.524 |  |  | — |
| SGD | 4 | lr 0.01, width 2048 | 1 | 1.000 | 0.581 | 0.587 | 0.581 |  |  | — |
| Adam | 8 | defaults | 5 | 0.100 | 0.100 | 0.535 | 0.101 | 0.000 | 0.60 | 310, 300, 330, 370, 330 |
| Adam | 8 | lr 3e-4 | 3 | 0.995 | 0.540 | 0.548 | 0.574 | 0.606 | 0.54 | — ×3 |
| Adam | 8 | lr 1e-4 | 3 | 0.998 | 0.557 | 0.559 | 0.589 | 0.637 | 0.58 | — ×3 |
| Adam | 8 | ε 1e-12 | 3 | 0.103 | 0.103 | 0.533 | 0.108 | 0.000 | 0.59 | 330, 440, 430 |
| Adam | 8 | ε 1e-4 | 3 | 0.102 | 0.103 | 0.545 | 0.107 | 0.000 | 0.59 | 370, 430, 430 |
| Adam | 8 | cosine 150 | 3 | 0.997 | 0.530 | 0.537 | 0.569 | 0.598 | 0.70 | — ×3 |
| Adam | 8 | cosine 500 | 3 | 0.993 | 0.523 | 0.541 | 0.571 | 0.576 | 0.56 | — ×3 |
| AdamW | 8 | weight decay 0.01 | 3 | 0.968 | 0.528 | 0.540 | 0.573 | 0.586 | 0.45 | — ×3 |
| SGD | 8 | lr 0.01 | 3 | 1.000 | 0.518 | 0.559 | 0.570 | 0.572 | 0.71 | — ×3 |
| SGD | 8 | lr 0.05 | 3 | 1.000 | 0.571 | 0.572 | 0.600 | 0.669 | 0.66 | — ×3 |
| Adam | 16 | defaults | 5 | 0.243 | 0.182 | 0.526 | 0.496 | 0.044 | 0.36 | 350, 380, n, —, 240 |
| Adam | 16 | data order ≠ init | 4 | 0.100 | 0.100 | 0.439 | 0.100 | 0.167 | 0.67 | n, n, 430, n |
| Adam | 16 | lr 3e-4 | 3 | 0.976 | 0.531 | 0.540 | 0.572 | 0.588 | 0.42 | — ×3 |
| Adam | 16 | lr 1e-4 | 3 | 0.993 | 0.538 | 0.545 | 0.577 | 0.605 | 0.49 | — ×3 |
| Adam | 16 | ε 1e-12 | 3 | 0.100 | 0.100 | 0.518 | 0.100 | 0.000 | 0.60 | n, 280, 490 |
| Adam | 16 | ε 1e-4 | 3 | 0.100 | 0.100 | 0.100 | 0.100 | 0.000 | 0.60 | n, n, n |
| Adam | 16 | cosine 150 | 3 | 0.828 | 0.467 | 0.496 | 0.526 | 0.482 | 0.65 | n, —, n |
| Adam | 16 | cosine 500 | 3 | 0.986 | 0.489 | 0.526 | 0.551 | 0.500 | 0.54 | — ×3 |
| Adam | 16 | batch 256 | 3 | 0.100 | 0.100 | 0.526 | 0.101 | 0.000 | 0.60 | 430, 460, 500 |
| Adam | 16 | batch 512 | 2 | 0.889 | 0.512 | 0.518 | 0.539 | 0.563 | 0.45 | — ×2 |
| Adam | 16 | width 256 | 3 | 0.384 | 0.311 | 0.516 | 0.495 | 0.157 | 0.11 | 370, —, n |
| Adam | 16 | width 2048 | 3 | 0.100 | 0.100 | 0.524 | 0.100 | 0.000 | 0.60 | 360, n, 280 |
| AdamW | 16 | weight decay 0.01 | 3 | 0.662 | 0.394 | 0.529 | 0.551 | 0.290 | 0.18 | 500, —, — |
| SGD | 16 | lr 0.01 | 3 | 0.100 | 0.100 | 0.100 | 0.100 | 0.000 | 0.60 | n, n, n |
| SGD | 16 | lr 0.01, data order ≠ init | 4 | 0.100 | 0.100 | 0.100 | 0.100 | 0.167 | 0.67 | n ×4 |
| SGD | 16 | lr 0.05 | 3 | 0.100 | 0.100 | 0.100 | 0.100 | 0.000 | 0.60 | n, n, n |
| SGD | 16 | lr 0.01, width 256 | 1 | 0.100 | 0.100 | 0.100 | 0.100 |  |  | n |
| SGD | 16 | lr 0.01, width 2048 | 1 | 0.100 | 0.100 | 0.100 | 0.100 |  |  | n |

**Column definitions:**

- Accuracies are means over seeds.
- **Peak test acc:** the highest test accuracy over checkpoints, averaged over seeds.
- **Seed-avg test acc:** the accuracy of the seed-averaged outputs at the end.
- **Pairwise test agreement:** the fraction of test images on which two seeds predict the same class, averaged over pairs.
- **Learning-time rank correlation:** the Spearman correlation between seeds of the checkpoint after which each training image stays correct.

Per-checkpoint curves for every config are in rich\_summary.json and compact.json.

## Training-dynamics measurements

**Two headline numbers:**

- **Preconditioned top eigenvalue:** from epoch 50 on, in lr-10⁻³ Adam runs that had not collapsed, it was between 4.9×10⁴ and 1.7×10¹². The 38/lr reference is 3.8×10⁴.
- **ε-dominated coordinates:** at epoch 150 the fraction was 0.18, 0.33, 0.47 and 0.74 at depths 2, 4, 8 and 16.

All values are means over seeds, and the seed counts are as in the main grid. The 13 deep checkpoints are listed in What was run.

**Top Hessian eigenvalue H / top eigenvalue of P⁻¹H**, by epoch. The training loss is on 2,048 fixed images. The H values at epoch 0 for Adam defaults were 70.7, 4.37, 1.51 and 1.50 at depths 2, 4, 8 and 16. References: 38/lr = 38,000 at lr 10⁻³ and 380,000 at lr 10⁻⁴. For SGD with momentum 0.9, 2(1+β)/lr = 380 at lr 0.01.

| Run | Epoch 10 | 50 | 150 | 300 | 500 |
| --- | --- | --- | --- | --- | --- |
| Adam, 2 layers | 29.1 / 4,750 | 69.0 / 76,500 | 177 / 159,000 | 248 / 77,300 | 271 / 109,000 |
| Adam, 4 layers | 6.62 / 33,000 | 89.7 / 183,000 | 250 / 6.46e6 | 497 / 3.06e7 | 1,350 / 1.26e8 |
| Adam, 8 layers | 2.59 / 60,900 | 20.9 / 269,000 | 370 / 1.84e7 | 11,700 / 3.47e6 | 1.65 / 81.1 |
| Adam, 16 layers | 8.45 / 255,000 | 300 / 168,000 | 166 / 284,000 | 177 / 154,000 | 213 / 2,240 |
| Adam, 8 layers, cosine 500 | 2.72 / 72,800 | 19.8 / 562,000 | 138 / 265,000 | 418 / 49,300 | 492 / 250,000 |
| Adam, 16 layers, cosine 500 | 26.7 / 67,300 | 26.2 / 77,100 | 386 / 259,000 | 98.0 / 60,000 | 216 / 266,000 |
| Adam, 16 layers, lr 1e-4 | 11.8 / 38,500 | 192 / 180,000 | 119 / 165,000 | 216 / 441,000 | 24.6 / 69,900 |
| Adam, 16 layers, ε 1e-4 | 2.94 / 2,210 | 1.00 / 37.4 | 1.00 / 37.8 | 1.00 / 37.6 | 1.00 / 37.6 |
| Adam, 8 layers, ε 1e-12 | 2.85 / 2.67e8 | 30.2 / 7.58e9 | 224 / 1.71e12 | 2,450 / 59,000 | 1.88 / 129 |
| SGD lr 0.01, 4 layers | 31.8 | 19.4 | 23.4 | 24.3 | 24.0 |
| SGD lr 0.01, 16 layers | 1.37 | 1.22 | 1.15 | 1.00 | 1.00 |

**Fraction of Adam coordinates with √v̂ < ε**, by epoch:

| Run | Epoch 10 | 50 | 150 | 300 | 500 |
| --- | --- | --- | --- | --- | --- |
| Adam, 2 layers | 0.0004 | 0.086 | 0.179 | 0.189 | 0.197 |
| Adam, 4 layers | 0.011 | 0.195 | 0.332 | 0.359 | 0.374 |
| Adam, 8 layers | 0.146 | 0.387 | 0.465 | 0.477 | 0.925 |
| Adam, 16 layers | 0.345 | 0.681 | 0.738 | 0.793 | 0.956 |
| Adam, 8 layers, cosine 500 | 0.149 | 0.419 | 0.482 | 0.505 | 0.572 |
| Adam, 16 layers, cosine 500 | 0.357 | 0.704 | 0.742 | 0.760 | 0.768 |
| Adam, 16 layers, lr 1e-4 | 0.212 | 0.340 | 0.358 | 0.357 | 0.343 |
| Adam, 16 layers, ε 1e-4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Adam, 8 layers, ε 1e-12 | 0.063 | 0.079 | 0.447 | 0.494 | 0.677 |

**Empirical NTK on 500 probe images** (250 training, 250 test). Label alignment is the centered kernel alignment with the labels on the 250 training images. Change is ‖K\_t − K\_0‖/‖K\_0‖. Kernel regression trains on the 250 and is scored on the 250 test images.

| Run | Alignment ep 0 / 150 / 500 | Change ep 150 / 500 | Kernel-regression acc ep 0 / 150 / 500 |
| --- | --- | --- | --- |
| Adam, 2 layers | 0.158 / 0.473 / 0.527 | 4.37 / 6.91 | 0.292 / 0.423 / 0.442 |
| Adam, 4 layers | 0.169 / 0.727 / 0.699 | 53.5 / 191 | 0.290 / 0.514 / 0.482 |
| Adam, 8 layers | 0.154 / 0.564 / 0.002 | 39.4 / 0.63 | 0.238 / 0.501 / 0.068 |
| Adam, 16 layers | 0.037 / 0.263 / 0.021 | 6.86 / 27.5 | 0.113 / 0.450 / 0.150 |
| Adam, 8 layers, cosine 500 | 0.153 / 0.646 / 0.527 | 18.3 / 51.3 | 0.248 / 0.499 / 0.523 |
| Adam, 16 layers, cosine 500 | 0.025 / 0.302 / 0.353 | 11.6 / 191 | 0.103 / 0.392 / 0.508 |
| Adam, 16 layers, lr 1e-4 | 0.025 / 0.089 / 0.122 | 78.0 / 41.4 | 0.103 / 0.552 / 0.551 |
| Adam, 16 layers, ε 1e-4 | 0.025 / 0.001 / 0.001 | 0.31 / 0.31 | 0.103 / 0.068 / 0.068 |
| Adam, 8 layers, ε 1e-12 | 0.153 / 0.640 / 0.019 | 23.9 / 0.69 | 0.248 / 0.512 / 0.075 |
| SGD lr 0.01, 4 layers | 0.168 / 0.424 / 0.426 | 4.65 / 4.95 | 0.285 / 0.567 / 0.547 |
| SGD lr 0.01, 16 layers | 0.025 / 0.003 / 0.001 | 0.25 / 0.31 | 0.103 / 0.128 / 0.068 |

**Representations at epoch 500** (1,000 test images). Inactive fraction is the share of ReLU units that output 0 on every probe image. Effective rank is exp(entropy) of the normalized squared singular values. “Middle” is layer 2 of 4, layer 4 of 8, or layer 8 of 16.

| Run | Inactive: first hidden layer | Inactive: middle | Inactive: last hidden layer | Effective rank, last hidden layer | Effective rank, 10 outputs |
| --- | --- | --- | --- | --- | --- |
| Adam, 2 layers | 0.002 | — | 0.697 | 23.2 | 9.48 |
| Adam, 4 layers | 0.000 | 0.387 | 0.966 | 9.97 | 9.74 |
| Adam, 8 layers | 0.000 | 0.546 | 0.997 | 1.11 | 1.01 |
| Adam, 16 layers | 0.000 | 0.944 | 0.986 | 2.16 | 2.76 |
| Adam, 8 layers, cosine 500 | 0.000 | 0.358 | 0.874 | 9.27 | 9.90 |
| Adam, 16 layers, cosine 500 | 0.000 | 0.803 | 0.823 | 9.10 | 9.86 |
| Adam, 16 layers, lr 1e-4 | 0.000 | 0.159 | 0.308 | 9.33 | 9.91 |
| Adam, 16 layers, ε 1e-4 | 0.000 | 0.457 | 1.000 | 1.00 | 1.00 |
| Adam, 8 layers, ε 1e-12 | 0.000 | 0.444 | 0.997 | 1.03 | 1.05 |
| SGD lr 0.01, 4 layers | 0.000 | 0.000 | 0.003 | 13.7 | 9.47 |
| SGD lr 0.01, 16 layers | 0.000 | 0.457 | 1.000 | 1.00 | 1.00 |

The same quantities for every config and every deep checkpoint, plus weight norms, distances from init and per-neuron balance, are in rich\_summary.json.

## Init seed vs data-order seed

At depth 4, Adam runs agree on 54–55% of test predictions whether two runs share the init, share the data order, or share neither. SGD runs that share the init agree on 72%, against 67% otherwise.

The init seed sets the initial weights; the data-order seed sets the minibatch order. Agreement is the fraction of test images on which two runs predict the same class at epoch 500, averaged over pairs.

| Config | Same init, different order | Same order, different init | Both different |
| --- | --- | --- | --- |
| Adam, 4 layers, defaults | 0.545 (5 pairs) | 0.547 (5) | 0.540 (26) |
| SGD lr 0.01, 4 layers | 0.724 (5) | 0.666 (5) | 0.668 (11) |
| Adam, 16 layers, defaults | 0.000 (5) | 0.991 (5) | 0.032 (26) |
| SGD lr 0.01, 16 layers | 0.000 (5) | 1.000 (5) | 0.000 (11) |

In the depth-16 rows, most runs predict a single class at the end (see Measurement notes). The individual depth-16 Adam runs:

| Init seed | Data-order seed | Max train acc | Collapse epoch | Final test acc |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0.936 | 350 | 0.100 |
| 0 | 1 | 0.322 | never reached 0.8 | 0.100 |
| 0 | 2 | 0.684 | never reached 0.8 | 0.100 |
| 1 | 1 | 0.927 | 380 | 0.114 |
| 1 | 0 | 0.901 | 430 | 0.100 |
| 2 | 2 | 0.785 | never reached 0.8 | 0.100 |
| 2 | 0 | 0.406 | never reached 0.8 | 0.100 |
| 3 | 3 | 0.942 | did not collapse | 0.496 |
| 4 | 4 | 0.888 | 240 | 0.100 |

At depth 4, all 9 Adam runs reached 0.984–0.987 maximum training accuracy and ended at 0.499–0.517 test accuracy. The 7 SGD runs ended at 0.561–0.573. Agreement over time for every pair is in rich\_summary.json under data\_order.

## Weight-space measurements

In configs where no seed collapsed, seed-pair loss barriers were 0.36–2.03, with two exceptions at depth 4: width 2048 (10.95) and batch 64 (145). Among collapsed configs they were 0.000–0.004 at depth 16 and 0.61 to 1.5×10⁶ at depth 8.

- **Seed pairs:** seed 0 with seed 1, and seed 0 with seed 2; values are means of the two pairs.
- **Init → final:** seed 0 only.
- **Barrier:** the highest training loss on the line minus the higher endpoint. Training loss is 0.5 × squared error summed over outputs, averaged over a fixed 10,000-image training subset.
- **Accuracy dip:** the lower endpoint's test accuracy minus the lowest on the line.
- **Distance:** Euclidean distance over all weights and biases.
- **Hessian:** the top 3 of the 10 computed eigenvalues, seed 0, width 1024 and batch 128 runs only.

| Config | Seed-pair barrier | Seed-pair test acc dip | Seed-pair distance | Init → final barrier | Distance from init | Hessian top 3 |
| --- | --- | --- | --- | --- | --- | --- |
| Adam, 2 layers, defaults | 0.566 | 0.344 | 649 | 0.000 | 533 | 344, 344, 344 |
| Adam, 2 layers, lr 3e-4 | 0.434 | 0.369 | 286 | 0.000 | 206 | 58.5, 58.3, 58.3 |
| Adam, 2 layers, lr 1e-4 | 0.409 | 0.299 | 120 | 0.000 | 80 | 18.6, 18.6, 18.5 |
| Adam, 2 layers, ε 1e-12 | 0.535 | 0.341 | 656 | 0.000 | 537 | 282, 281, 281 |
| Adam, 2 layers, ε 1e-4 | 0.416 | 0.331 | 232 | 0.000 | 166 | 67.9, 67.6, 67.6 |
| Adam, 2 layers, cosine 150 | 0.463 | 0.310 | 260 | 0.000 | 192 | 41.2, 40.4, 40.1 |
| Adam, 2 layers, cosine 500 | 0.444 | 0.301 | 440 | 0.000 | 338 | 159, 158, 158 |
| AdamW, 2 layers | 0.459 | 0.330 | 380 | 0.003 | 306 |  |
| SGD lr 0.01, 2 layers | 0.396 | 0.263 | 46 | 0.000 | 19 | 51.0, 49.3, 44.8 |
| SGD lr 0.05, 2 layers | 0.438 | 0.365 | 68 | 0.000 | 40 | 3.63, 3.29, 2.96 |
| Adam, 4 layers, defaults | 2.030 | 0.404 | 986 | 0.018 | 730 | 629, 468, 388 |
| Adam, 4 layers, lr 3e-4 | 0.473 | 0.422 | 388 | 0.005 | 296 | 195, 149, 129 |
| Adam, 4 layers, lr 1e-4 | 0.446 | 0.411 | 156 | 0.000 | 107 | 121, 121, 119 |
| Adam, 4 layers, ε 1e-12 | 1.355 | 0.403 | 993 | 0.008 | 725 | 2,240, 1,490, 1,390 |
| Adam, 4 layers, ε 1e-4 | 0.469 | 0.378 | 334 | 0.000 | 248 | 185, 169, 134 |
| Adam, 4 layers, cosine 150 | 0.483 | 0.412 | 346 | 0.000 | 252 | 102, 78.1, 60.1 |
| Adam, 4 layers, cosine 500 | 0.479 | 0.391 | 624 | 0.000 | 459 | 168, 156, 143 |
| Adam, 4 layers, batch 64 | 145 | 0.132 | 1,485 | 89.0 | 1,060 |  |
| Adam, 4 layers, batch 256 | 0.685 | 0.415 | 720 | 0.013 | 536 |  |
| Adam, 4 layers, width 256 | 0.694 | 0.397 | 554 | 0.000 | 403 |  |
| Adam, 4 layers, width 2048 | 10.95 | 0.388 | 1,345 | 0.085 | 987 |  |
| AdamW, 4 layers | 0.499 | 0.399 | 536 | 0.000 | 398 |  |
| SGD lr 0.01, 4 layers | 0.432 | 0.384 | 58 | 0.000 | 17 | 24.6, 20.3, 19.1 |
| SGD lr 0.05, 4 layers | 0.447 | 0.423 | 73 | 0.000 | 36 | 6.81, 5.22, 4.99 |
| Adam, 8 layers, defaults | 19.9 | 0.024 | 1,020 | 0.081 | 727 | 1, 1, 1 |
| Adam, 8 layers, lr 3e-4 | 0.473 | 0.434 | 466 | 0.000 | 332 | 1,290, 946, 934 |
| Adam, 8 layers, lr 1e-4 | 0.460 | 0.456 | 196 | 0.000 | 130 | 16.0, 13.2, 12.1 |
| Adam, 8 layers, ε 1e-12 | 1.5e6 | 0.018 | 1,105 | 34,500 | 744 | 3.67, 3.63, 3.62 |
| Adam, 8 layers, ε 1e-4 | 0.608 | 0.032 | 718 | 0.000 | 487 | 1, 1, 1 |
| Adam, 8 layers, cosine 150 | 0.460 | 0.427 | 458 | 0.000 | 328 | 26.3, 20.4, 17.6 |
| Adam, 8 layers, cosine 500 | 0.468 | 0.421 | 798 | 0.000 | 577 | 1,070, 728, 237 |
| AdamW, 8 layers | 0.425 | 0.430 | 637 | 0.000 | 447 |  |
| SGD lr 0.01, 8 layers | 0.456 | 0.405 | 82 | 0.000 | 23 | 43.1, 42.2, 39.5 |
| SGD lr 0.05, 8 layers | 0.455 | 0.458 | 130 | 0.000 | 74 | 9.89, 7.24, 6.25 |
| Adam, 16 layers, defaults | 0.001 | 0.000 | 988 | 0.000 | 716 | 1, 1, 1 |
| Adam, 16 layers, lr 3e-4 | 0.451 | 0.430 | 499 | 0.000 | 355 | 182, 38.0, 21.4 |
| Adam, 16 layers, lr 1e-4 | 0.475 | 0.435 | 230 | 0.000 | 146 | 78.4, 13.6, 11.4 |
| Adam, 16 layers, ε 1e-12 | 0.001 | 0.001 | 900 | 0.000 | 487 | 1.38, 1.01, 1.01 |
| Adam, 16 layers, ε 1e-4 | 0.000 | 0.000 | 105 | 0.000 | 2 | 1, 1, 1 |
| Adam, 16 layers, cosine 150 | 0.361 | 0.370 | 428 | 0.035 | 305 | 43.1, 33.5, 32.3 |
| Adam, 16 layers, cosine 500 | 0.490 | 0.375 | 768 | 0.000 | 543 | 26.1, 14.0, 11.2 |
| Adam, 16 layers, batch 256 | 0.004 | 0.013 | 904 | 0.000 | 637 |  |
| Adam, 16 layers, batch 512 | 0.388 | 0.407 | 731 | 0.012 | 554 |  |
| Adam, 16 layers, width 256 | 0.164 | 0.130 | 596 | 0.000 | 433 |  |
| Adam, 16 layers, width 2048 | 0.003 | 0.015 | 1,140 | 0.000 | 899 |  |
| AdamW, 16 layers | 0.029 | 0.069 | 643 | 0.000 | 474 |  |
| SGD lr 0.01, 16 layers | 0.000 | 0.000 | 105 | 0.000 | 1 | 1, 1, 1 |
| SGD lr 0.05, 16 layers | 0.000 | 0.000 | 105 | 0.000 | 1 | 1, 1, 1 |

**Adam and SGD from the same initialization** (Adam defaults vs SGD, seeds 0–2; means over seeds, range in brackets):

| Hidden layers | Barrier, SGD lr 0.01 | Barrier, SGD lr 0.05 | Test acc dip, lr 0.01 / 0.05 | Distance |
| --- | --- | --- | --- | --- |
| 2 | 1.04 (0.79–1.29) | 0.74 (0.64–0.86) | 0.275 / 0.250 | 529–536 |
| 4 | 0.67 (0.55–0.73) | 0.77 (0.50–1.04) | 0.393 / 0.406 | 720–730 |
| 8 | 1.81 (0.17–4.95) | 0.68 (0.11–1.76) | 0.018 / 0.011 | 707–757 |
| 16 | 3e−5 (3e−5–4e−5) | 0 | 0.006 / 0.001 | 612–764 |

At depths 8 and 16, the Adam endpoints in this comparison are collapsed runs, and at depth 16 the SGD endpoints are at chance. Full loss and accuracy curves along every path are in post.json.

## Measurement notes

How some quantities were computed, where it changes how a number should be read.

- **Collapse** means: after training accuracy first reaches 0.8, the first checkpoint where it is below 0.2. "Never reached 0.8" is listed separately.
- **Collapsed runs predict one class** for every test image. Agreement between two collapsed runs is therefore 1.0 if they picked the same class and 0.0 otherwise. Their learning-time rank correlations (0.59–0.67) come mostly from ties: about 90% of training images are never learned.
- **Hessian top eigenvalues of exactly 1.0** appear in configs whose last hidden layer is 98.7–100% inactive at the end. With every unit of a layer inactive, only the output bias moves the loss, and the Hessian of 0.5 × mean squared error in that bias is the identity.
- **Seed counts.** Main-grid configs at the defaults have 5 seeds; most others have 3. Data-order runs (init seed ≠ data-order seed) are kept out of seed statistics and reported in their own section.
- **Interpolation paths** use 21 points (11 for init → final and epoch 150 → final). Loss is on a fixed 10,000-image training subset, accuracy on the full test set. Barrier = the highest loss on the path minus the higher endpoint.
- **Hessian:** top 10 eigenvalues of the training loss on 4,096 fixed training images, by Lanczos (scipy eigsh, tolerance 10⁻³), seed 0 only.
- **Preconditioned sharpness** is the top eigenvalue of P⁻¹H with P = (1 − β₁^t)(√v̂ + ε), taken from 30 power iterations on a fixed set of 2,048 training images (the same set for the unpreconditioned sharpness). The reference value 38/lr (Cohen et al. 2022) is 38,000 at lr 10⁻³.
- **ε-dominance** = the fraction of all Adam coordinates with √v̂ < ε, weighted by tensor size.
- **Precision.** Training in fp32 with TF32 off. Saved test outputs and final weights are stored in fp16.
