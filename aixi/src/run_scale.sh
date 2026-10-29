#!/bin/bash
# Experiment 4: scaling sweep. Needs runs/exp4_scale/pvec.npy (the planted prior).
cd "$(dirname "$0")/.."
( python3 src/train_scale.py --d 128 --pvec runs/exp4_scale/pvec.npy --out runs/exp4_scale/d128 ) 2>&1 | grep --line-buffered -v -i warning > runs/exp4_scale/queue_a.log &
( python3 src/train_scale.py --d 32 --pvec runs/exp4_scale/pvec.npy --out runs/exp4_scale/d32
  python3 src/train_scale.py --d 64 --pvec runs/exp4_scale/pvec.npy --out runs/exp4_scale/d64 ) 2>&1 | grep --line-buffered -v -i warning > runs/exp4_scale/queue_b.log &
wait
echo SCALE_DONE >> runs/exp4_scale/queue_a.log
