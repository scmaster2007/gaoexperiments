#!/bin/bash
# Experiment 1: two parallel queues (one per CPU core), one seed per queue, depths 1, 2, 4.
# Run from anywhere; paths are relative to the repository root.
cd "$(dirname "$0")/.."
mkdir -p runs/exp1_main
for seed in 0 1; do
  ( for L in 1 2 4; do
      python3 src/train.py --layers $L --seed $seed --steps 10000 --batch 128 --threads 1 --out runs/exp1_main/L${L}_s${seed} 2>&1 | grep -v Warning
    done ) > runs/exp1_main/queue_s${seed}.log &
done
wait
echo ALL_DONE >> runs/exp1_main/queue_s0.log
