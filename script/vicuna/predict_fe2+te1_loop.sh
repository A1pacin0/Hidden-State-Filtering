#!/bin/bash

for last_k_value in {1..8}; do

    for HARMFUL_TYPE in "Autodan" "gcg" "Deepinception" "ReNellm" "GPTFuzz" "ICA"; do
        DIR_HARMFUL="Vicuna-7b-v1.5_${HARMFUL_TYPE}+adv.safetensors" 
        CHECKPOINT= your_checkpoint 
        OUTPUT_FILE= your_output_file 

        # Debug output
        echo "Processing HARMFUL_TYPE: $HARMFUL_TYPE"
        echo "DIR_HARMFUL: $DIR_HARMFUL"
        echo "CHECKPOINT: $CHECKPOINT"
        echo "OUTPUT_FILE: $OUTPUT_FILE"

        # Run the Python script
        python predict_fe2+te1_vicuna.py \
            --dataset_harmful "$DIR_HARMFUL" \
            --checkpoint "$CHECKPOINT" \
            --last_k $last_k_value \
            --output_file "$OUTPUT_FILE" \
            --fixed_text "$HARMFUL_TYPE"
    done

    DIR_HARMLESS=
    OUTPUT_FILE_HARMLESS=

    # Debug output
    echo "Processing HARMLESS dataset"
    echo "DIR_HARMLESS: $DIR_HARMLESS"
    echo "CHECKPOINT: $CHECKPOINT"
    echo "OUTPUT_FILE_HARMLESS: $OUTPUT_FILE_HARMLESS"

    # Run the Python script for the harmless dataset
    python predict_fe2+te1_mistral.py \
        --dataset_harmless "$DIR_HARMLESS" \
        --checkpoint "$CHECKPOINT" \
        --last_k "$last_k_value" \
        --output_file "$OUTPUT_FILE_HARMLESS" \
        --fixed_text "Safe"

    # Process benchmark datasets
    for HARMFUL_TYPE in "AdvBench" "MaliciousInstruct"; do
        DIR_HARMFUL="Vicuna-7b-v1.5_${HARMFUL_TYPE}.safetensors"
        CHECKPOINT=
        OUTPUT_FILE=

        # Debug output
        echo "Processing HARMFUL_TYPE: $HARMFUL_TYPE"
        echo "DIR_HARMFUL: $DIR_HARMFUL"
        echo "CHECKPOINT: $CHECKPOINT"
        echo "OUTPUT_FILE: $OUTPUT_FILE"

        # Run the Python script
        python predict_fe2+te1_vicuna.py \
            --dataset_harmful "$DIR_HARMFUL" \
            --checkpoint "$CHECKPOINT" \
            --last_k $last_k_value \
            --output_file "$OUTPUT_FILE" \
            --fixed_text "$HARMFUL_TYPE"
    done
done
