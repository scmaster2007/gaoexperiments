# Results: the NTK under Adam vs SGD

## Setup

- **Data and model:** CIFAR-10, all 50,000 training and 10,000 test images, flattened standardized pixels. ReLU MLP
  with biases, width 1024, 2, 4 or 8 hidden layers, PyTorch default init, squared loss on one-hot labels, batch 128,
  500 epochs. Data, init and data order as in stage 1 (`rich.py`).
- **Optimizers:** Adam with lr 10⁻³, β (0.9, 0.999), ε 10⁻⁸; SGD with lr 0.01, momentum 0.9. Constant learning rate throughout all of the tests. 
- **Seeds:** 0, 1, 2. 18 runs, which took about 62 minutes on one H100.
- **Kernels**, on 500 fixed probe images (250 train, 250 test; the same as stage 1), at epochs 0, 1, 2, 3, 5, 7, then
  every 10 epochs to 500 (56 checkpoints):
  - NTK: (1/10) Σ_k J_k J_kᵀ over the 10 outputs, with J_k the Jacobian of output k for all weights and biases.
  - Adam kernel (Adam runs): (1/10) Σ_k J_k diag(1/(√v̂ + ε)) J_kᵀ, with v̂ Adam's bias-corrected second moment at
    that step. It is the kernel of one Adam step with the momentum replaced by the current gradient and the
    preconditioner held fixed. It is undefined at step 0.
  - Adam kernel with √v̂ < ε removed: the same with those coordinates set to 0.
- **Hessian:** top eigenvalue of the training loss on 2048 fixed training examples (power iteration) at epochs 0, 1, 3,
  10, 50, 100, 150, 200, 250, 300, 400, 500.

## Measurements

- **Change from reference:** ‖K(t) − K(ref)‖ / ‖K(ref)‖ (Frobenius norm). The reference is epoch 0 for the NTK and
  epoch 1 for the two Adam kernels.
- **Kernel scaled to norm 1:** the same measurement with each K divided by its Frobenius norm, so only the shape counts.
- **Change per epoch:** ‖K(t) − K(previous checkpoint)‖ / ‖K(previous checkpoint)‖, divided by the epochs between them.
- **Epoch where change reaches 90%:** the first checkpoint where the change from reference is at least 0.9 × its
  final value.
- **Path length ÷ net change:** the sum of the changes between consecutive checkpoints divided by
  ‖K(final) − K(ref)‖. 1 means a straight line.
- **Label alignment:** centered alignment of the 250 × 250 train block with the one-hot labels.
- **Kernel-regression accuracy:** test accuracy of kernel regression from the 250 train probes to the 250 test probes
  (ridge 10⁻⁶ × the mean diagonal).
- **Direction changes:** sign changes between consecutive steps of a curve, ignoring steps smaller than 0.002
  (alignment) or 0.004 (accuracy).
- **Total variation ÷ net change:** the sum of the absolute steps of a curve divided by |last − first|. 1 means monotone.

In the epochs 0–200 table, "final" means epoch 200 and "last 100 epochs" means epochs 100–200. That table uses 25–26
checkpoints per run instead of 56, so its direction-change counts and path ratios are not comparable with the
epochs 0–500 table.

## Adam at depth 8

All 3 Adam depth-8 runs collapsed to predicting one class. Training accuracy peaked at 0.963, 0.964 and 0.969
(seeds 0, 1, 2) and first fell below 0.2 between epochs 300 and 350 (seeds 0 and 1) and between epochs 350 and 400
(seed 2). Stage 1 recorded the same at these settings (5 of 5 seeds, epochs 300–370). The epochs 0–500 values for
Adam at depth 8 include the collapse.

## Epochs 0–500

| Measurement (depth 2 / 4 / 8) | NTK, SGD | NTK, Adam | Adam kernel | Adam kernel, √v̂ < ε removed |
|---|---|---|---|---|
| Epoch where change from reference reaches 90% of its final value | 22.3 (7–40) / 20 / 137 | 287 / 437 / 37.3 (1–110) | 183 (60–430) / 15.7 / 2 | 53.3 / 30 / 2 |
| Same, kernel scaled to norm 1 | 26.7 / 36.7 / 20 | 40 / 23.3 / 1 | 46.7 / 8 / 2 | 33.3 (10–50) / 15.7 (7–30) / 2.33 |
| Change per epoch, mean over the last 100 epochs | 0.0026 / 0.0043 / 0.0062 | 0.038 / 0.07 / 0.012 | 0.25 / 0.75 (0.22–1.67) / 0.0083 (0.00068–0.017) | 0.14 / 0.28 / 0.0083 (0.00068–0.017) |
| Same, kernel scaled to norm 1 | 0.0025 / 0.0043 / 0.0062 | 0.038 / 0.066 / 1.4×10⁻⁵ | 0.12 / 0.13 / 1.4×10⁻⁵ (1.8×10⁻¹⁶–4.2×10⁻⁵) | 0.093 / 0.12 / 1.4×10⁻⁵ (1.8×10⁻¹⁶–4.2×10⁻⁵) |
| Path length ÷ net change | 3.3 / 5.47 / 8.52 | 15.1 / 14.9 / 4.0×10⁵ (1,416–1.1×10⁶) | 84.4 (10.2–140) / 2.9×10⁴ (2,804–7.5×10⁴) / 7.5×10⁴ (4,446–2.2×10⁵) | 80.2 / 422 / 1.2×10⁴ (1,229–2.9×10⁴) |
| Same, kernel scaled to norm 1 | 4.04 / 4.57 / 14.4 | 18 / 28.4 / 1.4×10⁴ (357–2.1×10⁴) | 42.6 / 56.8 / 48.3 | 41.1 / 55.3 / 50 |
| Direction changes, label alignment | 1 / 1 / 8.67 | 32.7 / 30.3 / 26.3 | 37.3 / 38.7 / 28.7 | 32 / 33.3 / 29.3 |
| Direction changes, kernel-regression accuracy | 27.3 / 24.3 / 26 | 32 / 30.3 / 21.7 | 33.7 / 32 / 22.3 | 33.7 / 32 / 22.3 |
| Total variation ÷ net change, label alignment | 1.16 / 1.09 / 1.45 | 2.73 / 3.86 / 42.9 | 255 (38.8–497) / 37.2 / 35.7 | 15.3 / 35.7 / 29.9 |
| Total variation ÷ net change, kernel-regression accuracy | 2.72 / 1.77 / 2.14 | 8.16 / 7.43 / 14.8 | 15.2 / 10.1 / 7.66 | 16.2 / 10 / 7.54 |
| Final change from reference | 0.92 / 4.67 / 30.8 | 5.32 / 205 / 4.17 (0.053–12.1) | 145 (21.4–392) / 3.88 / 0.95 | 5.77 / 4.81 / 0.95 |
| Final change from reference, kernel scaled to norm 1 | 1.02 / 1.14 / 0.93 | 1.31 / 1.2 / 0.036 (0.0016–0.1) | 1.36 / 1.22 / 1.09 | 1.19 / 1.19 / 1.01 |
| Final label alignment | 0.26 / 0.43 / 0.73 | 0.51 / 0.69 / 0.0034 | 0.32 / 0.58 / 0.0046 (0.00047–0.012) | 0.51 / 0.58 / 0.0046 (0.00047–0.012) |
| Final kernel-regression accuracy | 0.49 / 0.56 / 0.55 | 0.4 / 0.47 / 0.1 | 0.4 / 0.49 / 0.1 | 0.4 / 0.49 / 0.1 |

## Epochs 0–200 (before the Adam depth-8 collapse)

| Measurement (depth 2 / 4 / 8) | NTK, SGD | NTK, Adam | Adam kernel | Adam kernel, √v̂ < ε removed |
|---|---|---|---|---|
| Epoch where change from reference reaches 90% of its final value | 17.3 (2–30) / 20 / 157 | 153 / 173 / 180 | 117 (60–200) / 93.3 (40–200) / 90 | 53.3 / 96.7 (40–200) / 110 |
| Same, kernel scaled to norm 1 | 26.7 / 33.3 / 20 | 33.3 / 26.7 / 100 | 50 / 15.7 / 12.3 | 33.3 (10–50) / 33.3 (10–60) / 26.7 |
| Change per epoch, mean over the last 100 epochs | 0.0066 / 0.0057 / 0.023 | 0.044 / 0.066 / 0.12 | 0.17 / 1.34 (0.15–3.24) / 4.39 (0.25–12.4) | 0.11 / 0.31 (0.18–0.56) / 1.76 (0.24–4.47) |
| Same, kernel scaled to norm 1 | 0.0065 / 0.0056 / 0.021 | 0.042 / 0.063 / 0.088 | 0.11 / 0.14 / 0.14 | 0.094 / 0.13 / 0.14 |
| Path length ÷ net change | 2.53 / 4.13 / 5.42 | 6.09 / 6.6 / 4.56 | 18.5 (4.52–34.2) / 52.1 (1.27–112) / 2,421 (8.62–7,171) | 39.2 (18.1–63.7) / 32.1 (2.47–47.1) / 232 (6.31–683) |
| Same, kernel scaled to norm 1 | 3.07 / 3.37 / 12.6 | 9.34 / 12 / 13.1 | 17.3 / 22.2 / 22.1 | 18.2 / 21.3 / 21.3 |
| Direction changes, label alignment | 1 / 0.67 / 7 (3–10) | 12.3 / 9 / 13.7 | 14.7 / 15.7 / 14.7 | 13 / 12.7 / 15.3 |
| Direction changes, kernel-regression accuracy | 10.3 / 8.67 / 11 | 12 / 12.7 / 11.7 | 12.3 / 12 / 12.3 | 12.3 / 12 / 13 |
| Total variation ÷ net change, label alignment | 1.1 / 1.03 / 1.44 | 1.56 / 1.73 / 79.2 (4.02–225) | 19.4 / 44.8 / 43.6 (4.73–95) | 6.16 / 22.3 / 34.9 (5.38–75.6) |
| Total variation ÷ net change, kernel-regression accuracy | 1.74 / 1.29 / 1.71 | 3.18 / 2.95 / 2.9 | 5.76 / 3.4 / 3.48 | 6.08 / 3.43 / 3.57 |
| Final change from reference | 0.91 / 4.54 / 34.4 | 3.8 / 69.8 / 116 (56–209) | 98.5 (26.1–196) / 4,020 (9.12–1.2×10⁴) / 45.2 | 5.25 / 159 (6.58–455) / 57.9 (24–81) |
| Final change from reference, kernel scaled to norm 1 | 1.02 / 1.13 / 0.9 | 1.3 / 1.25 / 1.32 | 1.38 / 1.35 / 1.4 | 1.19 / 1.3 / 1.39 |
| Final label alignment | 0.26 / 0.43 / 0.7 | 0.5 / 0.67 / 0.37 (0.1–0.63) | 0.2 (0.04–0.49) / 0.17 (0.012–0.4) / 0.32 (0.014–0.78) | 0.48 / 0.32 (0.015–0.54) / 0.32 (0.014–0.78) |
| Final kernel-regression accuracy | 0.5 / 0.56 / 0.55 | 0.45 / 0.52 / 0.51 | 0.43 / 0.52 / 0.52 | 0.43 / 0.52 / 0.52 |

## Per run

| Optimizer | Depth | Seed | Final train acc | Final test acc | Peak test acc (20 logged epochs) | Top Hessian eigenvalue at epochs 0 / 200 / 500 | Fraction √v̂ < ε at epoch 500 | Minutes |
|---|---|---|---|---|---|---|---|---|
| SGD | 2 | 0 | 1.000 | 0.525 | 0.551 | 68.3 / 49.45 / 51.45 | – | 12 |
| SGD | 2 | 1 | 1.000 | 0.536 | 0.558 | 72 / 45.83 / 49.43 | – | 12 |
| SGD | 2 | 2 | 1.000 | 0.533 | 0.563 | 64 / 49.03 / 50.25 | – | 14 |
| SGD | 4 | 0 | 1.000 | 0.568 | 0.578 | 4.02 / 23.94 / 23.66 | – | 18 |
| SGD | 4 | 1 | 1.000 | 0.570 | 0.576 | 5.05 / 23.49 / 23.7 | – | 18 |
| SGD | 4 | 2 | 1.000 | 0.567 | 0.571 | 3.91 / 24.23 / 24.07 | – | 19 |
| SGD | 8 | 0 | 1.000 | 0.525 | 0.551 | 1.54 / 53.55 / 39.03 | – | 29 |
| SGD | 8 | 1 | 1.000 | 0.512 | 0.552 | 1.49 / 49.51 / 37.57 | – | 29 |
| SGD | 8 | 2 | 1.000 | 0.514 | 0.552 | 1.5 / 73.97 / 64.67 | – | 21 |
| Adam | 2 | 0 | 0.956 | 0.475 | 0.514 | 68.3 / 169.9 / 226.2 | 0.201 | 24 |
| Adam | 2 | 1 | 0.959 | 0.471 | 0.512 | 72 / 242.4 / 332.9 | 0.192 | 24 |
| Adam | 2 | 2 | 0.958 | 0.472 | 0.516 | 64 / 223.9 / 343.6 | 0.198 | 23 |
| Adam | 4 | 0 | 0.948 | 0.510 | 0.526 | 4.02 / 298.2 / 1554 | 0.379 | 35 |
| Adam | 4 | 1 | 0.963 | 0.509 | 0.531 | 5.05 / 331.1 / 3805 | 0.384 | 34 |
| Adam | 4 | 2 | 0.968 | 0.508 | 0.528 | 3.91 / 257.6 / 967.8 | 0.372 | 30 |
| Adam | 8 | 0 | 0.100 | 0.100 | 0.537 | 1.54 / 375.3 / 1 | 1.000 | 53 |
| Adam | 8 | 1 | 0.100 | 0.100 | 0.531 | 1.49 / 1681 / 1.363 | 1.000 | 40 |
| Adam | 8 | 2 | 0.101 | 0.101 | 0.535 | 1.5 / 544.4 / 19.05 | 1.000 | 27 |
