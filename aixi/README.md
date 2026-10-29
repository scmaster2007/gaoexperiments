# AIXI Experiments

How far is a trained network's prior from an exact Solomonoff-style prior?

I enumerate every program up to 13 instructions on a tiny Brainfuck-like machine (5 instructions, a 10-bit tape, 4-bit inputs). Each program gets weight 5^-length, and summing over programs that compute the same 16-entry truth table gives an exact prior over 5,545 behaviours. Small transformers are trained to predict outputs of sequences drawn from this prior, so the exact Bayes predictor is the optimum of the training loss, and I can compare the network's implied prior q(b) with the true P(b) for every behaviour.

## Results so far

1. The networks get close to Bayes-optimal overall (0.003–0.011 bits per prediction of excess loss), but the per-behaviour gap is large and systematic: behaviours with short programs (length 6–9) are underweighted by 4–6 bits, and complex ones (length 12–13) are overweighted by 2.5–5 bits. The gaps agree across seeds (r = 0.85–0.94).
2. Behaviours whose programs run longer are underweighted, even after controlling for prior mass and simple statistical features (partial r of about -0.2; a shuffled placebo gives about 0). None of the training changes I tried (importance sampling, soft targets, SAM, an exchangeability penalty) removed this at matched compute.
3. When rare behaviours are planted in the training data, the network's belief barely moves until about 1,000 exposures, and its belief about a behaviour it has never seen predicts its belief after training (r = 0.73–0.81).
4. That exposure threshold falls with model size, roughly as params^-0.4: from about 1,400 exposures at 26k parameters to about 280 at 398k. This is three sizes with one seed each, so the exponents are first estimates.

## Layout

```
enumeration/   C program that builds the exact prior
data/          the exact priors (one row per behaviour)
src/           training, analysis and plotting scripts
runs/          raw outputs of each training run, one folder per experiment
results/       summary tables
figures/
notes/         full write-up: setup, all tables, and every command to reproduce
```

To rerun an analysis, run it from this `aixi/` folder, for example `python3 src/analyze.py`. The full setup, results tables and reproduce commands are in `notes/details.md`.
