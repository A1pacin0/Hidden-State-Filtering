import os
import json
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed,AutoConfig
import torch
import logging
from tqdm import tqdm
import warnings
from utils import patch_open, logging_cuda_memory_usage
from safetensors.torch import save_file
import gc
import random
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def prepend_sys_prompt(sentence, args):
    messages = [{'role': 'user', 'content': sentence.strip()}]
    return messages


def forward(model, toker, messages):
    input_text = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(
        toker.convert_tokens_to_ids(toker.tokenize(input_text)),
        dtype=torch.long,
    ).unsqueeze(0).to(model.device)

    outputs = model(
        input_ids,
        attention_mask=input_ids.new_ones(input_ids.size(), dtype=model.dtype),
        return_dict=True,
        output_hidden_states=True,
    )
    hidden_states = [e[0].detach().half().cpu() for e in outputs.hidden_states[1:]]

    return hidden_states

def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--use_harmless", action="store_true")
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default='./hidden_states')
    parser.add_argument("--low_mem_mode",action="store_true")
    parser.add_argument("--reverse",action="store_true")
    args = parser.parse_args()



    # prepare model
    model_name = args.model_name = args.pretrained_model_path.split('/')[-1]

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        # use_safetensors=True,
        device_map="auto",
        # attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else None,
    )
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    num_layers = config.num_hidden_layers

    logging.info(f"Model name: {model_name}")
    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    
    logging_cuda_memory_usage()

    # prepare toker
    toker = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast='Orca-2-' not in model_name)


    # prepare data
    if args.use_harmless :
        data_path = './data_harmless'
        args.output_path += "_harmless"
    else:
        data_path = './data'
    
    input_file_path = args.input_file_path
    file_name_with_extension = os.path.basename(input_file_path)
    file_name, extension = os.path.splitext(file_name_with_extension)
    if extension == '.txt':
        with open(input_file_path) as f:
            lines = f.readlines()

    os.makedirs(f"{args.output_path}", exist_ok=True)

    # prepend sys prompt
    all_queries = [e.strip() for e in lines]
    n_queries = len(all_queries)

    all_messages = [prepend_sys_prompt(l, args) for l in all_queries]


    
    # tensors = {}
    # for idx, messages in tqdm(enumerate(all_messages),
    #                           total=len(all_messages), dynamic_ncols=True):
    #     hidden_states = forward(model, toker, messages)
    #     last_hidden_state = hidden_states[num_layers - 1]
    #     tensors[f'sample.{idx}_layer.{num_layers - 1}'] = last_hidden_state  # shape == (seq_len,hidden_size)

    def process_message(idx, message, model, tokenizer, num_layers):
        hidden_states = forward(model, tokenizer, message)
        last_hidden_state = hidden_states[num_layers - 1]
        return idx, last_hidden_state
    
    def save_hidden_states(model, tokenizer, messages, num_layers, model_name, file_name):
        tensors = {}
        processed_count = 0 
        
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_message, idx, message, model, tokenizer, num_layers)
                for idx, message in enumerate(messages)
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
                idx, last_hidden_state = future.result()
                # print(idx)
                tensors[f'sample.{idx}_layer.{num_layers - 1}'] = last_hidden_state
                processed_count += 1
                 # Memory management
                # logging_cuda_memory_usage()
                if processed_count % 300 == 0: 
                    logging_cuda_memory_usage()
                    torch.cuda.empty_cache()
                    gc.collect()

        save_file(tensors, f'{args.output_path}/{model_name}_{file_name}.safetensors')
    if args.low_mem_mode:
        logging.info(f"Running")
        tensors = {}
        processed_count = 0 
        for idx, messages in tqdm(enumerate(all_messages),
                              total=len(all_messages), dynamic_ncols=True):
            hidden_states = forward(model, toker, messages)
            last_hidden_state = hidden_states[num_layers - 1]
            tensors[f'sample.{idx}_layer.{num_layers - 1}'] = last_hidden_state # shape == (seq_len,hidden_size)
            processed_count += 1
            if processed_count % 100 == 0: 
                    logging_cuda_memory_usage()
                    torch.cuda.empty_cache()
                    gc.collect()
            
        if args.reverse:
            os.makedirs(f"{args.output_path}/reverse", exist_ok=True)
            save_file(tensors, f'{args.output_path}/reverse/{model_name}_{file_name}.safetensors')
        else:    
            save_file(tensors, f'{args.output_path}/{model_name}_{file_name}.safetensors')
    else: 
        logging.info(f"Running")
        save_hidden_states(model, toker, all_messages, num_layers, model_name, file_name)

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
