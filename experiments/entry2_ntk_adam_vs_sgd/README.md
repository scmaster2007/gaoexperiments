# Entry 2: Adam eNTK vs SGD eNTK

We use the same stage 1 setup to measure how the empirical neural tangent kernel changes during training under Adam vs SGD. We continue to use CIFAR-10 ReLU MLPs, with width 1024 and depths 2/4/8 and 500 epochs. I personally ran this on a cluster on November 1st. The "results.md" file has the setup, the file descriptions, and all of the actual data values too.

## Figures

- `figures/kernels.png`: change per epoch, change from the reference checkpoint, shape change per epoch, label
  alignment and kernel-regression accuracy against epoch, one column per depth. Line = mean of 3 seeds, band = min–max.
- `figures/hessian_accuracy.png`: top Hessian eigenvalue of the training loss and test accuracy against epoch.

## Data

Kernel names: `K_std` is the NTK, `K_adam` the Adam kernel J·diag(1/(√v̂+ε))·Jᵀ, `K_live` the Adam kernel with
the coordinates where √v̂ < ε removed (definitions in `results.md`).

- `data/runs_compact.json`: per run, every scalar measurement, plus the curves at 20 epochs (0, 1, 2, 3, 5, 7, 10,
  20, 30, 50, 70, 100, 150, ..., 500) rounded to 4 significant figures. Per run: `curves` (epoch, train/test accuracy
  and loss, fraction of coordinates with √v̂ < ε), `dead_frac` (dead units per hidden layer), `hessian`
  ([epoch, top eigenvalue, dominant eigenvalue]), and `kernels` → `K_std` / `K_adam` / `K_live` → scalars and curves
  (`epoch`, `norm`, `chg_ref`, `shp_ref`, `spd`, `shp_spd`, `align`, `krr`).
- `data/window_0_200.json`: the same measurements on epochs 0–200 only. `summary` → `<opt>_L<depth>` → `kernels` →
  one value per seed; `curves` has accuracy, loss and Hessian every 10 epochs to epoch 200 for the Adam depth-8 runs.
- `data/summary_table.json`: seed mean, min and max of each scalar (written by `plot.py`).

## Code

- `kern.py`: one run. Same model, data, init, data order, loss and optimizer settings as `../stage1_cifar_mlp/rich.py`;
  saves the kernels at each checkpoint (see its docstring). Imports `core.py` from `../stage1_cifar_mlp`.
- `kern_analyze.py`: the measurements in `results.md`, for a folder of runs.
- `kern_window.py`: the same, using only checkpoints up to a given epoch.
- `compact.py`: `data/runs_compact.json` from `kern_analyze.py`'s output.
- `plot.py`: the figures and `data/summary_table.json`.
- `test_kern.py`: checks `kern.py`'s kernels against brute-force per-example gradients (CPU, tiny sizes):
  `python test_kern.py [DATA_DIR]`.
- `jobs.txt`: the 18 runs, one command per line.

To reproduce, run the lines of `jobs.txt` from this folder, then:

```
python kern_analyze.py ~/kern ~/data ~/kern/results.json
python compact.py ~/kern/results.json data/runs_compact.json
python kern_window.py ~/kern ~/data 200 data/window_0_200.json
python plot.py data/runs_compact.json figures
```

## Raw data

The kernel checkpoints (`ck_*.npz`, about 2 GB, 56 per run) and the full `results.json` (447 KB) are not in the
repo. They are on the disk of the GPU machine the runs used (`sungwoo-adamfn-h1`).
