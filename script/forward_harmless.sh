#!/bin/bash


full_model_names=("your_model_path")

for full_model_name in ${full_model_names[@]}; do

model=${HF_MODELS}/${full_model_name}

python forward.py \
    --use_harmless \
    --low_mem_mode \
    --input_file_path  \
    --pretrained_model_path ${model}

done
