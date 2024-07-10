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
    return data


def get_acc_prob(data, target_name, draft_name):
    for item in data:
        target_log_p = eval(item['log_p_' + target_name])
        target_log_p = torch.tensor(target_log_p)

        draft_log_p = eval(item['log_p_' + draft_name])
        draft_log_p = torch.tensor(draft_log_p)

        diff = target_log_p - draft_log_p
        diff[diff>0] = 0
        acc_prob = torch.exp(diff)
        item[f'p_acc'] = acc_prob.tolist()

    return data

def parse_args():
    parser = argparse.ArgumentParser(description='data generator')

    parser.add_argument('--target_name', type=str, choices=["7b", "13b", "70b"], default='70b')
    parser.add_argument('--draft_name', type=str, choices=["7b", "13b", "70b"], default='7b')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default=None)

    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    data = read_data(args.input_file)
    data = get_acc_prob(data, target_name = args.target_name, draft_name = args.draft_name)

    if args.output_file is None or len(args.output_file) == 0:
        args.output_file = args.input_file.rstrip('.json') + '_acc.json'

    data = pretty_format(data)
    with open(args.output_file, 'w') as f:
        f.write(json.dumps(data, indent=2))

