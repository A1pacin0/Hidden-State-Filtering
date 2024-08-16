#!/bin/bash

for last_kth_value in {1..8}; do
    
    python train_fe2+te1_llama.py \
        --dataset_harmful  \
        --dataset_harmless \
        --checkpoint \
        --last_kth $last_kth_value

   
done
