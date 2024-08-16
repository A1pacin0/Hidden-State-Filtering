#!/bin/bash


full_model_names=("your_model_path")

for full_model_name in ${full_model_names[@]}; do

model=${HF_MODELS}/${full_model_name}

    for HARMFUL_TYPE in "Autodan" "gcg" "Deepinception" "ReNellm" "GPTFuzz" "ICA"; do
    python forward.py \
        --low_mem_mode \
        --input_file_path "data/model_name/${HARMFUL_TYPE}+adv.txt" \
        --pretrained_model_path ${model}

    done
    




done
