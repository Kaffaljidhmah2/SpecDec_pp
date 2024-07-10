"""example code for sampling using SpecDec++"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import torch
from hf_generation import my_generate
from wrap_model import AcceptancePredictionHead

device = "cuda"



def set_up():
    checkpoint = "meta-llama/Llama-2-70b-chat-hf"
    assistant_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
    assist_acc_head_dir = "hacky/acchead-llama2-chat-7bx70b"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint, torch_dtype=torch.bfloat16, device_map='cuda:0')
    assist_acc_head = AcceptancePredictionHead.from_pretrained(assist_acc_head_dir).to('cuda:0')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map='auto')


    return tokenizer, model, assistant_model, assist_acc_head


def format_prompt(prompt):
    """
        wrap the prompt in llama-2-chat format.
    """

    B_INST, E_INST = "[INST]", "[/INST]"
    return f"{B_INST} {prompt.strip()} {E_INST}"



def main(prompt):

    ### load target/draft/Acceptance Head and set generation config
    tokenizer, model, assistant_model, assist_acc_head = set_up()
    stop_threshold = 0.7
    bound = (2, 20)
    max_length = 512

    before=time.time()

    ### format and tokenize prompt
    prompt = format_prompt(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs, mismatched_tokens, LM_call = my_generate(model=model, **inputs, assistant_model=assistant_model, \
            max_length=max_length, num_assistant_tokens_schedule='ada', \
            do_sample=True, \
            assist_acc_head=assist_acc_head, \
            stop_threshold=stop_threshold, bound=bound)

    after = time.time()
    assisted_time = (after - before)

    print(tokenizer.decode(outputs[0]))

    print("assisted time: {:.2f}".format(assisted_time))
    print("# mismatched_tokens: {:.2f}".format(mismatched_tokens))
    print("# LM_call: {:.2f}".format(LM_call))

    return outputs



if __name__ == "__main__":
    prompt = "List 10 methods to be a successful PHD."
    main(prompt)

