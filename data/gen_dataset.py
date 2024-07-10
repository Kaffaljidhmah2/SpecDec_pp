from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json
import torch
from datasets import load_dataset, load_from_disk
import os
import argparse
from tqdm import tqdm
from util import CKPT, get_model, get_dataset, pretty_format

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

def get_prompt(sample, dataset_name):
    """
        wrap the prompt in llama-2-chat format.
    """
    if dataset_name == 'tatsu-lab/alpaca':
        prompt = get_prompt_alpaca(sample)
    elif dataset_name == 'openai_humaneval':
        prompt = get_prompt_humaneval(sample)
    elif dataset_name == 'gsm8k_test':
        prompt = get_prompt_gsm8k(sample)

    return f"{B_INST} {prompt.strip()} {E_INST}"


def get_prompt_alpaca(sample):
    """
        for alpaca format only
    """
    if sample['input'] is None or len(sample['input'].strip()) == 0:
        prompt = sample['instruction']
    else:
        prompt = sample['instruction'] + '\nInput: ' + sample['input']
    return prompt
    

def get_prompt_humaneval(sample):
    """
        OpenAI HumanEval
        prompt format https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/humaneval_gen.py
    """
    INSTRUCTION = """Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{prompt}

### Response:"""
    return INSTRUCTION.format(prompt=sample['prompt'])

def get_prompt_gsm8k(sample):
    """
        gsm8K
        prompt format https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    """
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    return problem_prompt.format(instruction=sample['question'])


def sanity_check(sample, tokenizer):
    inputs = tokenizer(get_prompt(sample), return_tensors='pt')
    input_ids = inputs['input_ids']
    assert input_ids[0][0] == tokenizer.bos_token_id, 'the first should be <bos>'
    assert input_ids[0][-1] != tokenizer.eos_token_id, 'the last should not be <eos>'
    print("sanity check passed")


def infer(prompt, tokenizer, model, max_length = 32, do_sample=False):

    device = "cuda"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_length += len(inputs['input_ids'][0]) 
    model_output = model.generate(**inputs, do_sample=do_sample, max_length = max_length)[0]

    ret = tokenizer.decode(model_output, skip_special_tokens=True)


    # Is it possible that this is not reversible? after decoding, the tokens changed? Yes!

    #re_token = tokenizer.encode(ret, return_tensors='pt').to(device)
    #if re_token.shape != model_output.shape or  (re_token - model_output).sum().item() != 0:
        #print("mismatch!")

    prefix_token_id =  model_output[:len(inputs['input_ids'][0])]
    gen_token_id = model_output[len(inputs['input_ids'][0]) :]
    ret = ret[len(prompt):] # remove prefix

    return  prefix_token_id.cpu().tolist() , gen_token_id.cpu().tolist(), ret



def parse_args():
    parser = argparse.ArgumentParser(description='data generator')

    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str, choices=["7b", "13b", "70b"])
    parser.add_argument('--mode', type=str, choices=['hf']) 
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--n_begin', type=int, default=0)
    parser.add_argument('--n_end', type=int, default=-1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_file', type=str, default=None)



    args = parser.parse_args()

    return args



def main(args):
    #sanity_check(dataset[0], tokenizer)

    print(f"we are using do sample = {args.do_sample}")

    tokenizer, model = get_model(args.model_name)
    dataset = get_dataset(args.dataset_name)

    if args.n_end == -1:
        args.n_end = len(dataset)
    args.n_end = min(args.n_end, len(dataset))

    res_dict = []
    for i in tqdm(range(args.n_begin, args.n_end)):
        sample = dataset[i]
        prompt = get_prompt(sample, args.dataset_name)
        prefix_token, gen_token, s = infer(prompt, tokenizer, model, max_length=args.max_length, do_sample=args.do_sample)
        res_dict.append(
            {
                'prompt': prompt,
                'continuation': s,
                'prefix': str(prefix_token) if prefix_token is not None else "",
                'tokens': str(gen_token) if gen_token is not None else ""
            }
        )

    if args.output_file is None:
        args.output_file = f'dataset{args.n_begin}to{args.n_end}_{args.mode}{args.model_name}.json'
    with open(args.output_file, 'w') as f:
        f.write(json.dumps(res_dict, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
