#!/bin/bash

cd ../../heart_negatives

python create_heart_negatives.py --dataset ogbl-collab --eps 1e-5 --use-val-in-test