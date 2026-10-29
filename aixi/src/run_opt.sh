#!/bin/bash
# Experiment 2: one queue per CPU core (one seed each). All runs: 2 layers, 5000 steps, batch 128, same LR schedule.
cd "$(dirname "$0")/.."
mkdir -p runs/exp2_optimizer
for seed in 0 1; do
  ( python3 src/train_opt.py --seed $seed --steps 5000 --out runs/exp2_optimizer/base_s$seed
    python3 src/train_opt.py --seed $seed --steps 5000 --is_tau 0.6 --out runs/exp2_optimizer/is_s$seed
    python3 src/train_opt.py --seed $seed --steps 5000 --target soft --out runs/exp2_optimizer/soft_s$seed
    python3 src/train_opt.py --seed $seed --steps 5000 --exch 0.1 --out runs/exp2_optimizer/exch_s$seed
    python3 src/train_opt.py --seed $seed --steps 5000 --opt sam --out runs/exp2_optimizer/sam_s$seed
  ) 2>&1 | grep --line-buffered -v -i warning > runs/exp2_optimizer/queue_s$seed.log &
done
wait
echo OPT_DONE >> runs/exp2_optimizer/queue_s0.log
