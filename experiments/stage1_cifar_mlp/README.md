# Stage 1: Adam vs SGD on deep MLPs, CIFAR-10, as well as some other data+

this folder contains all of the GPU runs from Oct 29–31, 2026. ReLU MLPs (no normalization, PyTorch default init) trained on full CIFAR-10
with squared loss for 500 epochs, using Adam, AdamW and SGD at depths 2 to 16. I think this took around 50 dollars? Not sure. There are three sets of runs:

- the main grid: 173 runs varying depth, learning rate, ε, schedule, batch size, width and seeds
- full-batch Adam in the NTK parametrization at widths 256 to 8192 (71 runs)
- checks against known NTK and signal-propagation results (R1–R4)

`results.md` has the setup and all result tables. `explorer.html` is an interactive viewer for the same numbers
(open it in a browser). Its data is in `data/explorer_data.json`, described in `data/README.md`.

## Code

- `core.py`: CIFAR-10 loading, MLPs, empirical and infinite-width NTK
- `rich.py`: one main-grid run, with the diagnostics listed in its docstring
- `adamwide.py`: one Adam-at-width run
- `repro.py`: R1–R4 and the epoch pilot
- `post.py`: loss along straight lines between solutions, Hessian spectra
- `sum_rich.py`, `sum_aw.py`, `analyze.py`: summaries of the three sets
- `export_web.py`, `compact.py`: compact extracts of the summaries
- `mkrich.py`, `mkextra.py`: write the job lists
- `test_core.py`: checks for `core.py`

`jobs/` lists every run as one command per line; run them from this folder. The scripts read CIFAR-10 from
`~/data` (change with `--data`, or `CIFAR` for the summary scripts).

## Raw data

The raw files (about 25 GB) and the full summary files are not in the repo. They are listed with their
artifact IDs under "Raw data" in `results.md`, and are stored until about Nov 30, 2026. Anything downloaded
goes in `downloads/` or `raw/`, which git ignores.
