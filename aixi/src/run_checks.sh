#!/bin/bash
# Experiment 1 checks: 3x longer training (machine A) and a second machine (machine B).
cd "$(dirname "$0")/.."
mkdir -p runs/exp1_long runs/exp1_machineB
# core 1: 3x longer training, same machine, 2 layers
( python3 src/train.py --layers 2 --seed 0 --steps 30000 --batch 128 --threads 1 --out runs/exp1_long/L2_s0 2>&1 | grep -v Warning ) > runs/exp1_long/log.txt &
# core 2: second machine, 2 layers, two seeds
( for s in 0 1; do python3 src/train.py --prior data/behaviours_machineB.csv --layers 2 --seed $s --steps 10000 --batch 128 --threads 1 --out runs/exp1_machineB/L2_s$s 2>&1 | grep -v Warning; done ) > runs/exp1_machineB/log.txt &
wait
echo CHECKS_DONE
