#!/bin/bash
# Experiment 3: wait for the optimizer runs to free the CPU cores, then train the 6 rotated planted-exposure models (2 at a time).
# Needs runs/exp3_plant/pvec_m*.npy (the planted priors).
cd "$(dirname "$0")/.."
while pgrep -f train_opt.py >/dev/null; do sleep 15; done
for half in 0 1; do
  ( for m in $(seq $half 2 5); do
      python3 src/train_plant.py --seed $m --steps 5000 --pvec runs/exp3_plant/pvec_m$m.npy --out runs/exp3_plant/m$m
    done ) 2>&1 | grep --line-buffered -v -i warning > runs/exp3_plant/queue_$half.log &
done
wait
echo PLANT_DONE >> runs/exp3_plant/queue_0.log
