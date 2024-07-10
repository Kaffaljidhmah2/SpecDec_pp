# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, TYPE_CHECKING, Any, Callable, Tuple, Union

import torch
import transformers
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import Trainer
import json
import numpy
import scipy.special
from ast import literal_eval as eval

from wrap_model import WrapModel, AcceptancePredictionHead
from transformers import EvalPrediction

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def compute_metrics(eval_pred: "EvalPrediction") -> Dict:
    num_class = 2
    logits= eval_pred[0]
    soft_labels = eval_pred[1]

    logits = logits.reshape(-1, num_class)
    soft_labels = soft_labels.reshape(-1)

    not_ignore = (soft_labels - IGNORE_INDEX) > 0.1

    target_prob = soft_labels[not_ignore]
    logits = logits[not_ignore]
    predicted_log_prob = scipy.special.log_softmax(logits, axis=-1)

    # KL divergence:
    CrossEnt = target_prob * ( - predicted_log_prob[:,1]) + (1-target_prob) * ( - predicted_log_prob[:,0])
    Ent = target_prob * numpy.log(target_prob) + (1-target_prob) * numpy.log(1-target_prob)
    Ent[numpy.isnan(Ent)] = 0.  # hack for binary entropy
    KL_binary = CrossEnt - Ent
    KL_binary = numpy.mean(KL_binary)

    return {'KL': KL_binary}


class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        soft_labels = inputs.pop('soft_labels')
        mask = (soft_labels - IGNORE_INDEX).abs() > 0.1

        soft_labels_1 = soft_labels
        soft_labels_0 = soft_labels_1.clone()
        soft_labels_0[mask] = 1 - soft_labels_1[mask]

        label_0 = torch.ones_like(soft_labels, dtype=torch.long).to(soft_labels.device) * IGNORE_INDEX
        label_0[mask] = 0
        label_1 = torch.ones_like(soft_labels, dtype=torch.long).to(soft_labels.device) * IGNORE_INDEX
        label_1[mask] = 1

        outputs = model.model(**inputs, output_hidden_states = True, return_dict=True)
        hidden_states = outputs.get("hidden_states")
        orignal_logits = model.assist_acc_head(hidden_states[-1])
        orignal_logits = orignal_logits.float()

        num_class = 2

        weight = torch.tensor([self.args.weight_mismatch, 1]).to(orignal_logits.device)
        loss_fct = CrossEntropyLoss(weight=weight, reduction='none')

        logits = orignal_logits.view(-1, num_class)
        label_0 = label_0.view(-1)
        label_1 = label_1.view(-1)
        soft_labels_0 = soft_labels_0.view(-1)
        soft_labels_1 = soft_labels_1.view(-1)
        mask = mask.view(-1)

        loss_0 = loss_fct(logits, label_0) # (bs * seq_len), num_class
        loss_1 = loss_fct(logits, label_1) # (bs * seq_len), num_class

        # reduce with soft labels, coresponding to BCELoss
        loss = (loss_0 * soft_labels_0 + loss_1 * soft_labels_1).sum() / (self.args.weight_mismatch * soft_labels_0[mask].sum() +  soft_labels_1[mask].sum() )

        if model.training:
            # KL divergence:
            target_prob = soft_labels_1[mask]
            predicted_logits = logits[mask, :]
            predicted_log_prob = torch.log_softmax(predicted_logits, dim=-1)

            #KL_binary = target_prob * (target_prob.log() - predicted_log_prob[:,1]) + (1-target_prob) * ( (1-target_prob).log() - predicted_log_prob[:,0])

            CrossEnt = target_prob * ( - predicted_log_prob[:,1]) + (1-target_prob) * ( - predicted_log_prob[:,0])
            Ent = target_prob * target_prob.log() + (1-target_prob) * (1-target_prob).log()
            Ent[Ent.isnan()] = 0.  # hack for binary entropy
            KL_binary = CrossEnt - Ent
            KL_binary = KL_binary.mean().item()

            self.log({'KL': KL_binary})


        if return_outputs:
            outputs = (loss, orignal_logits)
            return (loss, outputs)
        else:
            return loss

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bf16: bool = True
    model_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    remove_unused_columns: bool = False
    evaluate_only: bool = False
    label_names: Optional[List[str]] = field(
        default_factory=lambda: ['soft_labels'], metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    weight_mismatch: Optional[float] = field(default = 1.) # 6 for balancing classes
    resnet_num_layers: Optional[int] = field(default = 1)
    mixing_ratio: Optional[float] = field(default = 0.15)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, r: float = 0.15):
        super(SupervisedDataset, self).__init__()
        logging.warning(f"Loading data... from {data_path}")
        data = json.load(open(data_path,'r'))
        self.input_ids = []
        self.soft_labels = []
        for item in data:
            item['prefix'] = eval(item['prefix'])
            item['tokens'] = eval(item['tokens'])
            item['draft'] = eval(item['draft'])

            # item['tokens'] are generated autoregressively from target model
            # item['draft'] are stochatic next-token predicted by the draft model

            item['p_acc'] = eval(item['p_acc'])

            prefix = torch.LongTensor(item['prefix'])
            Xs = torch.LongTensor(item['tokens'])
            # Ys = torch.LongTensor(item['draft'])

            # take r from Xs and (1-r) from Ys.
            mask = (torch.rand(*Xs.shape) < r)
            Zs = torch.LongTensor(item['draft'])
            Zs[mask] = Xs[mask]

            self.input_ids.append(torch.cat([prefix, Zs]))

            label_prefix = torch.tensor([IGNORE_INDEX] * len(item['prefix']))
            p_acc = torch.tensor(item['p_acc'])

            # don't calculate loss on Xs.
            p_acc[mask] = IGNORE_INDEX

            self.soft_labels.append(torch.cat([label_prefix, p_acc]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], soft_labels=self.soft_labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, soft_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "soft_labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        soft_labels = torch.nn.utils.rnn.pad_sequence(soft_labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            soft_labels=soft_labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )




if __name__ == "__main__":
    parser = transformers.HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(training_args.model_name_or_path)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = SupervisedDataset(training_args.data_path, r=training_args.mixing_ratio)
    if training_args.eval_data_path is not None:
        eval_dataset = SupervisedDataset(training_args.eval_data_path, r=training_args.mixing_ratio)
        print("num eval example:", len(eval_dataset))
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    acc_head_config = {'hidden_size': model.config.hidden_size, 'num_layers': training_args.resnet_num_layers}
    assist_acc_head = AcceptancePredictionHead(acc_head_config)
    wrapped = WrapModel(model, assist_acc_head)
    wrapped.model.requires_grad_(False)
    print('num training example:', len(train_dataset))
    trainer = MyTrainer(model=wrapped, tokenizer=tokenizer, args=training_args, train_dataset = train_dataset, eval_dataset = eval_dataset, data_collator=data_collator, compute_metrics = compute_metrics)
    if training_args.evaluate_only:
        print("eval only. Loading from checkpoint:", training_args.output_dir)
        wrapped.assist_acc_head = AcceptancePredictionHead.from_pretrained(training_args.output_dir)
        trainer.evaluate()
    else:
        trainer.train()
        trainer.save_state()
        wrapped.assist_acc_head.save_pretrained(training_args.output_dir, config=acc_head_config)
