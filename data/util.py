from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json
import torch
from datasets import load_dataset, load_from_disk
import os
import argparse


CKPT = {
    '7b': "meta-llama/Llama-2-7b-chat-hf",
    '13b': "meta-llama/Llama-2-13b-chat-hf",
    '70b': "meta-llama/Llama-2-70b-chat-hf"
}


def get_model(model_name):
    checkpoint = CKPT[model_name]
    dtype = torch.bfloat16
    print('model checkpoint: ', checkpoint)
    print('model dtype: ', dtype)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer, model

def get_dataset(name):
    if name=="tatsu-lab/alpaca":
        dataset_file='alpaca'
        if not os.path.exists(dataset_file):
            dataset = load_dataset("tatsu-lab/alpaca")['train']
            dataset.save_to_disk(dataset_file)
        else:
            dataset = load_from_disk(dataset_file)
    elif name=='openai_humaneval':
        dataset_file='humaneval'
        if not os.path.exists(dataset_file):
            dataset = load_dataset('openai_humaneval')['test']
            dataset.save_to_disk(dataset_file)
        else:
            dataset = load_from_disk(dataset_file)
    elif name=='gsm8k_test':
        dataset_file='gsm8k'
        if not os.path.exists(dataset_file):
            dataset = load_dataset('gsm8k', 'main')['test']
            dataset.save_to_disk(dataset_file)
        else:
            dataset = load_from_disk(dataset_file)
    else:
        raise NotImplementedError
    return dataset

def pretty_format(data):
    for item in data:
        for key, value in item.items():
            if isinstance(value, list) and isinstance(value[0], int):
                item[key] = str(value)
            if isinstance(value, list) and isinstance(value[0], float):
                item[key] = str(value)
    return data

if __name__ == "__main__":
    dataset = get_dataset('gsm8k_test')
    print(len(dataset), dataset[0])