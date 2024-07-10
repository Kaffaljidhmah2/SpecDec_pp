from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json
import torch
from datasets import load_dataset, load_from_disk
import os
import argparse
from tqdm import tqdm
from ast import literal_eval as eval
from util import CKPT, get_model, pretty_format

def read_data(filename):
    data = json.load(open(filename,'r'))
    for item in data: 
        item['prefix'] = eval(item['prefix'])
        item['tokens'] = eval(item['tokens'])
        item['draft'] = eval(item['draft']) 
    return data


@torch.no_grad()
def get_log_prob(data, model, model_name):
    for item in data:
        joint = item['prefix'] + item['tokens']
        joint = torch.LongTensor(joint).to(model.device)
        joint = joint.unsqueeze(0)

        index = item['draft'] 
        index = torch.LongTensor(index).to(model.device)
        index = index.unsqueeze(-1)

        logits = model(input_ids = joint).logits
           
        log_probs = logits.log_softmax(dim=-1)  # bs * seq_len * vocab_size
        log_probs_shifted = log_probs[0, len(item['prefix'])-1 : -1] # next_token_log_prob for the continuation

        # take along "item['draft']" 
        log_p = torch.take_along_dim(log_probs_shifted, index, dim=-1) # seq_len * 1
        item[f'log_p_{model_name}'] = log_p[:, 0].tolist()

    return data

def parse_args():
    parser = argparse.ArgumentParser(description='data generator')

    parser.add_argument('--model_name', type=str, choices=["7b", "13b", "70b"], default='7b')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default=None)

    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    data = read_data(args.input_file)

    tokenizer, model = get_model(args.model_name)
    data = get_log_prob(data, model, args.model_name)
    suffix = 'logP'
    if args.output_file is None or len(args.output_file) == 0:
        args.output_file = args.input_file.rstrip('.json') + '_' + args.model_name + suffix + '.json'

    data = pretty_format(data)


    with open(args.output_file, 'w') as f:
        f.write(json.dumps(data, indent=2))

