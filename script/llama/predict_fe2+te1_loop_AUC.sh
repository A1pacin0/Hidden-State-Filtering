#!/bin/bash

for last_k_value in {1..8}; do

        CHECKPOINT=
        OUTPUT_FILE=
        # Run the Python script
        python predict_fe2+te1_llama_AUC.py \
            --dataset_harmful  \
            --dataset_harmless  \
            --checkpoint "$CHECKPOINT" \
            --last_k $last_k_value \
            --output_file "$OUTPUT_FILE" \
            --fixed_text "$HARMFUL_TYPE"

done