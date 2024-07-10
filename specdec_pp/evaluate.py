from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import time
from datetime import datetime
import torch
from hf_generation import my_generate
import argparse
import json
import os
import numpy as np
from ast import literal_eval as eval

device = "cuda"

def set_up(args):
    if args.do_sample:
        print("do_sample for SpeculativeDecoding")

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    checkpoint = args.model_name
    assistant_checkpoint = args.assistant_name

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint, torch_dtype=torch.bfloat16, device_map='cuda:0')

    if args.num_assistant_tokens_schedule == 'ada':
        from wrap_model import AcceptancePredictionHead
        print("Loading from acc_head checkpoint:", args.assist_acc_head_dir)

        assist_acc_head = AcceptancePredictionHead.from_pretrained(args.assist_acc_head_dir).to('cuda:0')

    else:
        assist_acc_head = None

    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map='auto')

    # print(model.hf_device_map)
    # print(assistant_model.hf_device_map)
    return model, assistant_model, tokenizer, assist_acc_head

def assist(model, assistant_model, tokenizer, assist_acc_head, inputs, max_length, num_assistant_tokens=None):
    # outputs = model.generate(**inputs, generation_config=generation_config, assistant_model=assistant_model, max_length=max_length)
    before=time.time()
    assistant_model.max_assistant_tokens = None
    outputs, mismatched_tokens, LM_call = my_generate(model=model, **inputs, assistant_model=assistant_model, \
            max_length=max_length, num_assistant_tokens_schedule=args.num_assistant_tokens_schedule, \
            num_assistant_tokens=num_assistant_tokens, do_sample=args.do_sample, \
            assist_acc_head=assist_acc_head, \
            stop_threshold=args.stop_threshold, bound=args.bound)

    after = time.time()
    assisted_time = (after - before)

    print("assisted time: {:.2f}".format(assisted_time))
    print("mismatched_tokens: {:.2f}".format(mismatched_tokens))
    print("LM_call: {:.2f}".format(LM_call))

    return outputs, mismatched_tokens, LM_call, assisted_time

def target(model, tokenizer, inputs, max_length):
    before=time.time()
    outputs = model.generate(**inputs, max_length=max_length, do_sample=args.do_sample)
    after = time.time()
    target_time = (after - before)
    print("target time {:.2f}".format(target_time))
    return outputs, target_time

def draft(assistant_model, tokenizer, inputs, max_length):
    before=time.time()
    outputs = assistant_model.generate(**inputs, max_length=max_length, do_sample=args.do_sample)
    after = time.time()
    draft_time = (after - before)
    print("draft time {:.2f}".format(draft_time))
    return outputs, draft_time


def run(model, assistant_model, tokenizer, assist_acc_head, args, item):
    len_prefix = len(eval(item['prefix']))
    inputs = {'input_ids': torch.LongTensor([eval(item['prefix'])]).to(device)}
    max_length = args.max_length
    print("max_length:", max_length)


    if args.num_assistant_tokens_schedule in ['constant', 'heuristic', 'ada']:

        if args.num_assistant_tokens_schedule == 'ada':
            num_assistant_tokens = None
        else:
            num_assistant_tokens = args.num_assistant_tokens
            print("num_assistant_tokens:", num_assistant_tokens)

        res_a, num_mismatched_tokens, num_LM_call, assisted_time = assist(model, assistant_model, tokenizer, assist_acc_head, inputs, max_length, num_assistant_tokens=num_assistant_tokens)
    elif args.num_assistant_tokens_schedule == 'none':
        res_a = [[-1]]
        num_mismatched_tokens = -1
        num_LM_call = -1
        assisted_time = -1
    else:
        raise ValueError(f"{args.num_assistant_tokens_schedule} not supported")


    if args.num_assistant_tokens_schedule == 'none':
        res_b, target_time = target(model, tokenizer, inputs, max_length)
        generated_length_target = len(res_b[0]) - len_prefix

        res_c, draft_time = draft(assistant_model, tokenizer, inputs, max_length)
        generated_length_draft = len(res_c[0]) - len_prefix
    else:
        target_time = -1
        generated_length_target = -1
        draft_time = -1
        generated_length_draft = -1



    generated_length = len(res_a[0]) - len_prefix
    print("generated_length: {:.2f}".format(generated_length))

    return assisted_time, target_time, draft_time, num_mismatched_tokens, num_LM_call, generated_length, generated_length_target, generated_length_draft

def parse_args():
    parser = argparse.ArgumentParser(description='benchmark performance')

    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--assistant_name', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--do_sample', action='store_true')

    parser.add_argument('--num_assistant_tokens', type=int, default=5)
    parser.add_argument('--num_assistant_tokens_schedule', type=str, default="constant", choices=['constant', 'heuristic', 'ada', 'none'])
    parser.add_argument('--assist_acc_head_dir', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='data/alpaca_data/test.json')
    parser.add_argument('--save_path', type=str, default='./test_results')
    parser.add_argument('--random_seed', type=int, default=47)
    parser.add_argument('--stop_threshold', type=float, default=None)
    parser.add_argument('--bound', nargs='+', type=int, default=None)

    parser.add_argument('--n_begin', type=int, default=0)
    parser.add_argument('--n_end', type=int, default=None)

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    args = parse_args()
    data = json.load(open(args.data_path,'r'))
    if args.n_end is None:
        args.n_end = len(data)
    args.n_end = min(len(data), args.n_end)


    os.makedirs(args.save_path, exist_ok=True)

    model, assistant_model, tokenizer, assist_acc_head = set_up(args)

    results = []

    for i, item in enumerate(data[args.n_begin:args.n_end]):
        print("---------------------------------")
        print(f"data {i + args.n_begin}")
        before=time.time()

        assisted_time, target_time, draft_time, num_mismatched_tokens, num_LM_call, generated_length, generated_length_target, generated_length_draft = run(model, assistant_model, tokenizer, assist_acc_head, args, item)
        item.update({
            'id': i+args.n_begin,
            'spec_time': assisted_time,
            'target_time': target_time,
            'draft_time': draft_time,
            'num_mismatched_tokens': num_mismatched_tokens,
            'num_LM_call': num_LM_call,
            'generated_length': generated_length,
            'generated_length_target': generated_length_target,
            'generated_length_draft': generated_length_draft,
        })
        results.append(item)

        after=time.time()
        print("total time: {:.2f}".format(after-before))
    save_file = f"{args.save_path}/results_{args.n_begin}to{args.n_end}.json"
    with open(save_file, 'w') as f:
        f.write(json.dumps(results, indent=2))
