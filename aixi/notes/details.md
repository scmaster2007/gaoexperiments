# Pilot: what prior does a trained network implement, against an exact Bayes reference?

## Setup

**Machine.** A small Brainfuck-family machine with five instructions: `<`, `>`, `+`, `[`, `]` (as in F-ICL's machine F).
- The tape has 10 bit cells. A 4-bit input is preloaded at cells 3–6, and the head starts at cell 3.
- The output is the bit under the head when the program halts. Machine B instead reads the bit in cell 3 at halt.
- A program is kept only if it halts within 128 steps and never leaves the tape, on all 16 inputs.

**Prior.** The enumeration covers every program of up to 13 instructions, about 41 million kept programs.
- A *behaviour* is the 16-entry truth table a program computes. There are 5,545 distinct behaviours on machine A and 2,002 on machine B.
- Each program gets prior weight 5^-|p|. P(behaviour) is the sum of these weights over all programs that compute it.

**Task.** Each training sequence is built as follows:
1. Sample a behaviour from P.
2. Show its 16 (input, output) pairs in a random order.
3. Ask the network to predict each output from the pairs before it.

The exact Bayes-optimal prediction at every position is computed from the enumeration. It is the optimum of the training objective.

**Networks.** Small causal transformers (d=64, 4 heads, 1, 2 or 4 layers).
- There are no positional embeddings, so the examples are exchangeable.
- Training uses 10,000 steps with batch size 128, so the network sees 1.28M sequences sampled freshly from P.

**Measurement.**
- The network's implied probability of each behaviour is computed by the chain rule over its 16 outputs. It is averaged over 16 random input orders.
- The *gap* of a behaviour is log2 q(b) − log2 P(b). This gap is the network's inductive bias relative to the exact prior.

## Layout

```
aixiexperiments/
├── README.md
├── enumeration/   C program that enumerates every program and builds the exact prior
├── data/          the exact prior: one row per behaviour (truth table)
├── src/           all Python and shell scripts (training, analysis, plotting)
├── runs/          raw outputs of each training run (weights, log-probabilities, logs)
│   ├── exp1_main/        Experiment 1: depths 1, 2, 4, two seeds
│   ├── exp1_long/        Experiment 1 check: 3x longer training
│   ├── exp1_machineB/    Experiment 1 check: second machine
│   ├── exp2_optimizer/   Experiment 2: training procedures
│   ├── exp3_plant/       Experiment 3: planted exposure (6 rotated models)
│   └── exp4_scale/       Experiment 4: scaling sweep (also holds the planted prior, law_data.npz)
├── results/       summary tables produced by the analysis scripts (`results/lm/` holds the Pythia follow-up)
└── figures/
```

| Path | What it is |
|---|---|
| `enumeration/enumerate.c` | Exhaustive enumeration. Add `-DOUTMODE=1` for machine B. `enumerate_linux` and `enumerate_b_linux` are Linux builds; on a Mac, recompile (see Reproduce). |
| `data/behaviours.csv` | Machine A prior: 5,545 behaviours from all programs up to 13 instructions, with counts per program length. |
| `data/behaviours_machineB.csv` | Machine B prior (2,002 behaviours). |
| `data/behaviours_L9.csv`, `_L12.csv`, `_L13.csv` | Earlier enumerations up to length 9, 12 and 13, without per-length counts. Not used by the scripts. |
| `src/common.py` | Prior loading, the exact Bayes predictor, the data sampler and the transformer. |
| `src/train.py`, `src/analyze.py` | Experiment 1: train one network; gap by program length, seed consistency, regressions, placebo test. |
| `src/train_opt.py`, `src/analyze_opt.py` | Experiment 2. |
| `src/train_plant.py`, `src/analyze_plant.py` | Experiment 3. |
| `src/train_scale.py`, `src/analyze_scale.py`, `src/plot_scale.py` | Experiment 4. |
| `src/run_*.sh` | The exact runs used for each experiment. |
| `src/build_data.py`, `src/lm_law.py`, `src/analyze_lm.py` | Follow-up on pretrained language models (Pythia), run on a GPU node. Only the summary tables in `results/lm/nodeB/` are included. |
| `src/train_loop*.py`, `src/train_scale_loop.py`, `src/lm_loops.py`, `src/analyze_loop.py`, `src/analyze_recur.py` | Follow-up on looped and recurrent-depth models, run on a GPU node. Raw outputs not included. |
| `results/results_*.csv` | Summary tables. No suffix = machine A; `_long` = the longer run; `_machineB` = machine B. |

## Reproduce

Run every command from the repository root.

```
gcc -O3 -march=native -o enumeration/enumerate enumeration/enumerate.c -lm && ./enumeration/enumerate 13 data/behaviours.csv
gcc -O3 -march=native -DOUTMODE=1 -o enumeration/enumerate_b enumeration/enumerate.c -lm && ./enumeration/enumerate_b 13 data/behaviours_machineB.csv
pip install torch numpy pandas scipy matplotlib
src/run_all.sh && python3 src/analyze.py
src/run_checks.sh && python3 src/analyze.py --runs runs/exp1_long --tag _long && python3 src/analyze.py --prior data/behaviours_machineB.csv --runs runs/exp1_machineB --tag _machineB
src/run_opt.sh && python3 src/analyze_opt.py
src/run_plant.sh && python3 src/analyze_plant.py
src/run_scale.sh && python3 src/analyze_scale.py && python3 src/plot_scale.py
```

On 2 CPU cores, enumeration takes about 1 minute per machine, and training takes 4–20 minutes per model.

## Results (October 14, 2026, 2 CPU cores)

The units throughout are bits. For each behaviour b, the gap is log2 q(b) − log2 P(b).

1. **The networks get close to the Bayes optimum overall.** Excess loss over the exact Bayes predictor, in bits per prediction:

   | Layers | Excess (mean of 2 seeds) |
   |---|---|
   | 1 | 0.0106 |
   | 2 | 0.0065 |
   | 4 | 0.0035 |
   | 2, trained 3× longer | 0.0032 |

2. **The remaining gap is large and systematic.** By shortest program length K:
   - Behaviours with K = 6–9 are underweighted by about 4–6 bits, a factor of 16–64.
   - Behaviours with K = 12–13 are overweighted by about 2.5–5 bits.

   Per-behaviour gaps agree across seeds, with Pearson r = 0.85, 0.92 and 0.94 for 1, 2 and 4 layers.

3. **Runtime predicts the gap beyond prior mass and six statistical-simplicity features.** Slower behaviours are underweighted. The partial correlation in each setting:

   | Setting | Partial r |
   |---|---|
   | Machine A, all depths | −0.19 to −0.29 |
   | Placebo (99th percentile of runtimes shuffled within each K) | 0.04 |
   | Machine A, 3× longer training | −0.27 |
   | Machine A, controlling for loop count | −0.09 to −0.20 |
   | Machine B | −0.10 to −0.11 (placebo 0.05) |

4. **Depth affects the two biases differently.**
   - The sensitivity bias shrinks with depth. Its standardised coefficient goes from about −1.86 at 1 layer to −1.17 at 4 layers.
   - The runtime bias does not. Its coefficient stays at about −1.2 for all depths, so the pilot does not support the prediction that more compute removes it.
   - A likely reason is the task: a network can do in-context induction over 16-entry truth tables without ever simulating a program.

5. **The networks are not coherent Bayesian predictors on rare behaviours.** The implied joint probability of a behaviour changes by about 3 bits depending on the order of the inputs. A Bayesian predictor would give the same value for every order.

Per-run coefficients are in `results/results_per_run_coefficients.csv`.

## Experiment 2: can a training procedure steer the network toward the exact posterior? (`src/train_opt.py`, `src/analyze_opt.py`)

All runs use 2 layers, 5,000 steps, batch 128 and 2 seeds. Compute is given relative to the 5k-step baseline.

| Procedure | Compute | Excess (bits/pred) | Mean abs gap, K 6–9 (bits) | Runtime coef. | Order dependence (bits) |
|---|---|---|---|---|---|
| Baseline (AdamW) | 1.0× | 0.0099 | 5.74 | −1.24 | 3.41 |
| Importance sampling (P^0.6, reweighted) | 1.1× | 0.0085 | 5.27 | −1.25 | 3.38 |
| Exact-posterior soft targets (oracle) | 1.9× | 0.0076 | 5.26 | −1.29 | 3.32 |
| SAM (ρ = 0.05) | 2.1× | 0.0106 | 4.78 | −1.01 | 3.10 |
| Exchangeability penalty (λ = 0.1) | 1.9× | 0.0791 | 4.19 | −1.32 | 0.30 |
| Baseline, 10k steps | 2.0× | 0.0065 | 4.92 | −1.25 | 3.19 |
| Baseline, 30k steps | 6.0× | 0.0032 | 4.67 | −1.46 | 2.96 |

- At matched compute, no procedure beats simply training the baseline longer.
- The runtime bias is unchanged by every procedure.
- The exchangeability penalty does make the network coherent: order dependence falls from 3.4 to 0.3 bits. But it does so by flattening the prior. Complex behaviours end up overweighted by about 9 bits, and the overall fit is 8× worse. Only one value of λ was tried.

## Experiment 3: planted exposure (`src/train_plant.py`, `src/analyze_plant.py`)

**Design.**
- 72 behaviours with K = 10–11 that the natural prior almost never shows. They were chosen to span the range of runtimes.
- They were split into 6 groups and planted at expected training exposures of 0, 4, 16, 64, 256 or 1,024 sequences out of 640k.
- 6 models were trained, rotating which group got which level, so every behaviour appears at every level. The exact Bayes reference is recomputed for each modified prior.

**Dose-response.** The network's belief about a planted behaviour, mean log2 q over 72 behaviours, against its true training probability:

| Exposures | log2 q (network) | log2 P (true) |
|---|---|---|
| 0 | −21.9 | — |
| 4 | −21.5 | −17.3 |
| 16 | −21.7 | −15.3 |
| 64 | −20.4 | −13.3 |
| 256 | −19.1 | −11.3 |
| 1,024 | −14.6 | −9.3 |

The belief barely moves until roughly 10^3 exposures.

**The zero-exposure belief predicts the learned belief.** A behaviour's never-seen belief, measured in a different model, predicts its belief at 4–256 exposures with r = 0.73–0.81. The between-model noise ceiling is r ≈ 0.73. The slope is about 0.7 up to 256 exposures and drops to 0.40 at 1,024.

**What sets the zero-exposure belief.**
- It tracks Boolean sensitivity (ρ = −0.48) and runtime (ρ = −0.25).
- It does not track the behaviour's algorithmic probability (ρ = 0.19, not significant) or K.
- The runtime effect fades as exposure grows: ρ = −0.25 at 0 exposures, −0.09 at 1,024.

**Exposure alone does not determine the belief.** At matched exposure, planted complex behaviours are 1.7–3.7 bits more underweighted than naturally common, simpler behaviours.

**Leave-one-behaviour-out prediction of log2 q(b, e).**

| Model of the network's belief | RMSE (bits) | R² |
|---|---|---|
| Network = exact Bayes | 7.3 | −1.56 |
| Exposure only | 3.8 | 0.31 |
| Exposure + zero-exposure belief | 2.5 | 0.70 |

## Experiment 4: scaling sweep for the exposure threshold (`src/train_scale.py`, `src/analyze_scale.py`, `src/plot_scale.py`)

**Design.**
- 195 complex behaviours (K = 10–11), stratified by runtime, were planted at fixed training frequencies: never, or 2^-19, 2^-17, 2^-15, 2^-13, 2^-11 or 2^-9 per sequence.
- Three model widths were trained (2 layers; d = 32, 64 and 128; 26k, 101k and 398k parameters), one seed each.
- Each run used 16k steps of batch 128, with a constant learning rate after warmup. An exponential moving average of the weights was evaluated at 128k, 256k, 512k, 1M and 2M training sequences.
- Exposure is frequency × sequences seen, so it ranges from 0.24 to 4,096.

**The fitted law.** For a behaviour b with exposure e:

    log2 q(b) ≈ D + (log2 P(b) − D) · S(e / e*),   S(x) = 1 / (1 + x^(−k)),   k ≈ 0.8

Here D is the network's belief about behaviours it has never seen, and e* is the threshold exposure.
- The curve's shape is about the same in every configuration. k ranges from 0.63 to 0.96, with one outlier at 0.41.
- Most of the transition from 10% to 90% happens over roughly a 250-fold range of exposures.

**How the threshold scales.**
- Fit to all 90 points at once: e* ∝ params^(−0.40 ± 0.06) · sequences^(−0.14 ± 0.08).
- Fit to the per-configuration thresholds, weighted by their standard errors: e* ∝ params^(−0.45) · sequences^(−0.21). This gives χ² = 13.2 on 11 degrees of freedom.
- Measured thresholds run from about 1,400 exposures (26k params) down to about 280 (398k params, 2M sequences).

**How well the curve fits.** R² is measured against a model in which beliefs never move from D.

| Model | R² | RMSE (bits) |
|---|---|---|
| One threshold for every configuration | 0.949 | 0.75 |
| Threshold depends on model size and data | 0.968 | 0.60 |

**Complication: the data term depends on model size.** With more training data:
- The 26k model's threshold stays flat (slope +0.05 ± 0.01).
- The 101k model's threshold falls (slope −0.27 ± 0.12).
- The 398k model's threshold falls (slope −0.29 ± 0.06).

Adding a size × data interaction lowers χ² from 13.2 to 7.4 for one extra parameter. A single product of power laws is therefore an approximation.

**Never-seen default.** For the larger models, D drops as training continues, from about −20 to −22 bits. They become more confident that unseen behaviours are unlikely.

**Limits.** Three sizes over a 16× range of parameters, one seed per size, one machine and one architecture family. The exponents are first estimates.

The figure is `figures/scaling_law.png`. Points are in `results/results_scale_points.csv`, thresholds in `results/results_scale_thresholds.csv`, and global fits in `results/results_scale_law.json`.
