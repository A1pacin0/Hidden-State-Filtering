#!/bin/bash

# 遍历 last_kth_value 从 1 到 8
for last_k_value in {1..8}; do
    # 调用 Python 脚本
    python /mnt/dataA/131/HSD/train_fe2+te1_vicuna.py \
        --dataset_harmful '/mnt/dataA/131/HSD/hidden_states/Vicuna-7b-v1.5_PKU-SafeRLHF-prompts.safetensors' '/mnt/dataA/131/HSD/hidden_states/Vicuna-7b-v1.5_UltraSafety.safetensors' \
        --dataset_harmless '/mnt/dataA/131/HSD/hidden_states_harmless/Vicuna-7b-v1.5_databricks-dolly-3k.safetensors' '/mnt/dataA/131/HSD/hidden_states_harmless/Vicuna-7b-v1.5_databricks-dolly-3_6k.safetensors'  \
        --checkpoint '/mnt/dataA/131/HSD/checkpoint/vicuna/last_k'\
        --last_k $last_k_value

done
