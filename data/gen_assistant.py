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

    return data


@torch.no_grad()
def get_assistant_result(data, assistant_model, model_name, do_sample):
    for item in data:
        joint = item['prefix'] + item['tokens']
        joint = torch.LongTensor(joint).to(assistant_model.device)
        joint = joint.unsqueeze(0)
        sm_logits = assistant_model(input_ids = joint).logits
        if do_sample:
            probs = sm_logits.softmax(dim=-1)  # bs * seq_len * vocab_size
            new_token = torch.multinomial(probs[0], num_samples=1).squeeze(-1)
            item['draft'] = new_token[len(item['prefix'])-1 : -1].tolist()
        else:
            new_token = sm_logits.argmax(dim=-1) # bs * seq_len
            item['draft'] = new_token[0, len(item['prefix'])-1 : -1].tolist()
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='data generator')

    parser.add_argument('--model_name', type=str, choices=["7b"], default='7b')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--do_sample', action='store_true')

    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    data = read_data(args.input_file)

    tokenizer, model = get_model(args.model_name)
    data = get_assistant_result(data, model, args.model_name, args.do_sample)

    if args.output_file is None or len(args.output_file) == 0:
        if args.do_sample:
            suffix = 'stochastic'
        else:
            suffix = 'greedy'
        args.output_file = args.input_file.rstrip('.json') + '_' + args.model_name + suffix + '.json'

    data = pretty_format(data)

    with open(args.output_file, 'w') as f:
        f.write(json.dumps(data, indent=2))

