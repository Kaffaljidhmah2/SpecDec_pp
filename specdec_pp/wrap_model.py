import transformers
from transformers import AutoTokenizer, EsmForMaskedLM,  AutoModelForCausalLM, Trainer, TrainingArguments
from tokenizers import Tokenizer
from dataclasses import dataclass, field
from typing import  Optional
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from huggingface_hub import PyTorchModelHubMixin

logger = logging.get_logger(__name__)


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        #torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

class AcceptancePredictionHead(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        self.config=config
        hidden_size = config['hidden_size']
        num_layers = config.get('num_layers', 0)
        super().__init__()
        self.model = nn.Sequential( *([ResBlock(hidden_size)] * num_layers), nn.Linear(hidden_size, 2) )

    def forward(self, x):
        return self.model(x)

class WrapModel(PreTrainedModel):
    def __init__(self, model, head):
        super().__init__(model.config)
        self.model = model
        self.assist_acc_head = head

    def forward(self, input_ids = None, labels = None, **kwargs):
        return self.model(input_ids = input_ids, labels = labels, **kwargs)


if __name__ == "__main__":
    #input_ids = labels = torch.LongTensor([[1,2,3]])
    #model = transformers.AutoModelForCausalLM.from_pretrained("ckpt/hf-llama2-7b-chat")
    #wrapped = WrapModel(model, num_layers=2)
    #AcceptancePredictionHead.from_pretrained('../exp-weight6-layer3')

    import pdb;pdb.set_trace()
