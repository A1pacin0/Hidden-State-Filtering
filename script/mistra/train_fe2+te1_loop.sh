#!/bin/bash


for last_k_value in {1..8}; do

    python train_fe2+te1_mistra.py \
        --dataset_harmful \
        --dataset_harmless \
        --checkpoint \
        --last_k $last_k_value




done
