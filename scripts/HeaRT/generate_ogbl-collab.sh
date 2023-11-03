#!/bin/bash

cd ../../heart_negatives

# When allowing train/valid samples to be negatives
# To **disallow them**, remove the  --keep-train-val flag
python create_heart_negatives.py --dataset ogbl-collab --eps 1e-5 --use-val-in-test --keep-train-val