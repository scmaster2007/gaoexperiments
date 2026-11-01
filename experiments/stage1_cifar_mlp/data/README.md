# explorer_data.json

Built by `../export_web.py` from the raw runs and summaries, then trimmed (per-layer CKA, weight norms and
per-layer distances removed). Numbers are rounded to 3–4 significant figures; NaN and infinity are `null`.

Run groups are named `{opt}_L{depth}_w{width}_b{batch}_lr{lr}_eps{eps}_{schedule}{epochs}`, with `_dataorder`
for runs whose data-order seed differs from the init seed.

`runs`: one entry per main-grid run.
- `cfg`, `seed`, `ds` (data-order seed), `e` (epochs of about 35 checkpoints)
- `tra`, `tea`: train and test accuracy × 1000; `trl`: train loss

`configs`: one entry per run group.
- `n`, `seeds`, `data_seeds`
- `collapse`: per seed, first epoch where train accuracy drops below 0.2 after reaching 0.8 (`null` if it
  did not collapse, "never reached 0.8" if it never got there)
- `tra_f`, `tra_max`, `tea_f`, `tea_max`: final and best train and test accuracy per seed
- `never`: fraction of training images never learned; `lt_rho`: rank correlation between seeds of the epoch
  each training image is learned
- `ce`, `savg`: checkpoint epochs and test accuracy of the seed-averaged outputs; `agree`: agreement between
  seeds' test predictions; `reldist`: relative distance from each seed's outputs to the seed average
- `de`: epochs of the detailed checkpoints, for the fields below
  - `sharp`: top Hessian eigenvalue; `psharp`: top eigenvalue of P⁻¹H (inflated where √v̂ < ε)
  - `ntk_align`, `ntk_change`, `ntk_acc`: NTK alignment with labels, relative change from init, and
    kernel-regression accuracy (250 train / 250 test points)
  - `dead`, `erank`: per layer, fraction of units inactive on all 1,000 probe images, and effective rank
- `epsf`, `epsf10`, `epse`: fraction of Adam coordinates with √v̂ below ε and below 10ε, at epochs `epse`

`data_order`: final test agreement and output distance for pairs of runs that share the init seed, share the
data order, or share neither.

`post`: measurements on saved weights.
- straight lines between two seeds, between Adam and SGD from the same init, from init to final weights, and
  from epoch 150 to the end; each has `b` (train-loss barrier), `dip` (test-accuracy dip), `trl` and `tea`
  along the line, and `dist` (weight distance)
- `hess`: top 10 Hessian eigenvalues

`adamwide`: Adam-at-width summaries as `{fields, rows}`. `repro`: R1–R4 summaries.
