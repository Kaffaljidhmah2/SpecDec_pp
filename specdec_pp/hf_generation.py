# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified from  https://raw.githubusercontent.com/huggingface/transformers/v4.34.1/src/transformers/generation/utils.py

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from transformers.utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

from transformers.generation.utils import GenerationMixin, GenerateOutput, GenerationMode, _crop_past_key_values, GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput


@torch.no_grad()
def my_generate(
    model: "PreTrainedModel",
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    num_assistant_tokens_schedule: Optional[str] = 'heuristic',
    num_assistant_tokens: Optional[int] = None,
    oracle_token_num_list: Optional[List[int]] = None,
    assist_acc_head: Optional[nn.Module] = None,
    stop_threshold: Optional[float] = None,
    bound: Optional[List[int]] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which had the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complement the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
            `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
            generating before other GPUs. Otherwise it'll be set to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchDecoderOnlyOutput`],
                - [`~generation.SampleDecoderOnlyOutput`],
                - [`~generation.BeamSearchDecoderOnlyOutput`],
                - [`~generation.BeamSampleDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
    """

    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    model._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
        # two conditions must be met
        # 1) the generation config must have been created from the model config (`_from_model_config` field);
        # 2) the generation config must have seen no modification since its creation (the hash is the same).
        if model.generation_config._from_model_config and model.generation_config._original_object_hash == hash(
            model.generation_config
        ):
            new_generation_config = GenerationConfig.from_model_config(model.config)
            if new_generation_config != model.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use and modify the model generation configuration (see"
                    " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                )
                model.generation_config = new_generation_config
        generation_config = model.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    model._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not model.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. determine generation mode
    generation_mode = model._get_generation_mode(generation_config, assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")

        # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
        if assistant_model.config.is_encoder_decoder:
            assistant_model_kwargs = copy.deepcopy(model_kwargs)
            inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
            )
            assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_model_kwargs, model_input_name
            )
            model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

        # 12. run assisted generate
        return my_assisted_decoding(
            model,
            input_ids,
            assistant_model=assistant_model,
            do_sample=generation_config.do_sample,
            logits_processor=logits_processor,
            logits_warper=model._get_logits_warper(generation_config) if generation_config.do_sample else None,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            num_assistant_tokens_schedule=num_assistant_tokens_schedule,
            num_assistant_tokens=num_assistant_tokens,
            oracle_token_num_list=oracle_token_num_list,
            assist_acc_head=assist_acc_head,
            stop_threshold=stop_threshold,
            bound=bound,
            **model_kwargs,
        )


def my_assisted_decoding(
    model: "PreTrainedModel",
    input_ids: torch.LongTensor,
    assistant_model: "PreTrainedModel",
    do_sample: bool = False,
    logits_processor: Optional[LogitsProcessorList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    num_assistant_tokens_schedule: Optional[str] = 'heuristic',
    num_assistant_tokens: Optional[int] = None,
    oracle_token_num_list: Optional[List[int]] = None,
    assist_acc_head: Optional[nn.Module] = None,
    stop_threshold: Optional[float] = None,
    bound: Optional[List[int]] = None,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
    **sample** (depending on `do_sample`), assisted by a smaller model. Can be used for text-decoder, text-to-text,
    speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.assisted_decoding`] directly. Use
    generate() instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> assistant_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id
    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
    >>> outputs = model.assisted_decoding(
    ...     input_ids,
    ...     assistant_model=assistant_model,
    ...     logits_processor=logits_processor,
    ...     stopping_criteria=stopping_criteria,
    ... )
    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""
    # Assistant: initialize assistant-related variables
    if num_assistant_tokens_schedule is None: # default to heuristic
        num_assistant_tokens_schedule = 'heuristic'

    if num_assistant_tokens is not None:
        assistant_model.max_assistant_tokens = num_assistant_tokens
        logger.warning("Setting initial assistant model max_assistant_tokens to %d" % num_assistant_tokens)
    else:
        if not hasattr(assistant_model, "max_assistant_tokens") or assistant_model.max_assistant_tokens is None:
            assistant_model.max_assistant_tokens = 5 # default to 5
        # this value, which will be updated if heuristic num_assistant_tokens_schedule is applied, persists across calls

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.generation_config.eos_token_id
    if eos_token_id is not None and pad_token_id is None:
        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else model.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else model.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else model.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    # other auxiliary variables
    max_len = stopping_criteria[0].max_length
    assistant_kv_indexing = (
        1
        if "bloom" in assistant_model.__class__.__name__.lower()
        or (
            assistant_model.config.architectures is not None
            and "bloom" in assistant_model.config.architectures[0].lower()
        )
        else 0
    )

    this_peer_finished = False  # used by synced_gpus only
    num_mismatched_tokens = 0
    assist_rounds = 0

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # Assistant: main logic start
        cur_len = input_ids.shape[-1]

        #  1. Forecast next N tokens using the assistant model. This `for` block can be replaced with a
        # `.generate()` call if we decide to add `past_key_values` as a possible output of generate, as we
        # need access to the assistant cache to secure strong speedups.
        candidate_input_ids = input_ids
        q_prob = []
        assist_steps = 0

        cum_acc_prob = 1. #  used for 'ada' schedule

        while True:
        # for _ in range(int(assistant_model.max_assistant_tokens)):
            if num_assistant_tokens_schedule != 'ada' and assist_steps >= assistant_model.max_assistant_tokens:
                break
            # 1.1. use the assistant model to obtain the next candidate logits
            if "assistant_past_key_values" in model_kwargs:
                prev_seq_len = model_kwargs["assistant_past_key_values"][0][assistant_kv_indexing].shape[-2]
                # `new_token_len` can be 1 or 2 (next token in assistant + last token picked by the larger model)
                new_token_len = candidate_input_ids.shape[1] - prev_seq_len
                assert new_token_len > 0, 'might have bug!'
                assist_inputs = candidate_input_ids[:, -new_token_len:]
                assist_attn = torch.ones_like(candidate_input_ids)
                # TODO (joao): make it compatible with models that use unconventional fwd pass logic, like blip2
                if assistant_model.config.is_encoder_decoder:
                    assistant_model_outputs = assistant_model(
                        decoder_input_ids=assist_inputs,
                        decoder_attention_mask=assist_attn,
                        past_key_values=model_kwargs["assistant_past_key_values"],
                        encoder_outputs=model_kwargs["assistant_encoder_outputs"],
                        output_hidden_states = True if num_assistant_tokens_schedule == 'ada' else False,
                    )
                else:
                    assistant_model_outputs = assistant_model(
                        assist_inputs,
                        attention_mask=assist_attn,
                        past_key_values=model_kwargs["assistant_past_key_values"],
                        output_hidden_states = True if num_assistant_tokens_schedule == 'ada' else False,
                    )
            else:
                if assistant_model.config.is_encoder_decoder:
                    assistant_model_outputs = assistant_model(
                        decoder_input_ids=candidate_input_ids,
                        encoder_outputs=model_kwargs["assistant_encoder_outputs"],
                        output_hidden_states = True if num_assistant_tokens_schedule == 'ada' else False,
                    )
                else:
                    assistant_model_outputs = assistant_model(candidate_input_ids,
                        output_hidden_states = True if num_assistant_tokens_schedule == 'ada' else False,
                        )


            model_kwargs["assistant_past_key_values"] = assistant_model_outputs.past_key_values
            if len(logits_processor) > 0:
                assistant_model_outputs.logits[:, -1, :] = logits_processor(
                    candidate_input_ids, assistant_model_outputs.logits[:, -1, :]
                )
            if len(logits_warper) > 0:
                assistant_model_outputs.logits[:, -1, :] = logits_warper(
                    candidate_input_ids, assistant_model_outputs.logits[:, -1, :]
                )

            # 1.2. greedily select the next candidate token; or do speculative decoding.
            if do_sample:
                probs = assistant_model_outputs.logits[:, -1, :].softmax(dim=-1)  # bs * vocab_size
                new_token = torch.multinomial(probs[0, :], num_samples=1)
                q_prob.append(probs)
            else:
                new_token = assistant_model_outputs.logits[:, -1, :].argmax(dim=-1)

            candidate_input_ids = torch.cat((candidate_input_ids, new_token[:, None]), dim=-1)

            # 1.3. stop assistant generation on EOS
            if eos_token_id_tensor is not None:
                last_assistant_token_is_eos = new_token.tile(eos_token_id_tensor.shape[0], 1)
                last_assistant_token_is_eos = (
                    ~last_assistant_token_is_eos.ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0).bool()
                )
                if last_assistant_token_is_eos:
                    break
            else:
                last_assistant_token_is_eos = False

            # 1.4. stop assistant generation when the max length is reached or when the stop head predcits stop
            assist_steps += 1
            
            # bound = (min, max) 
            if num_assistant_tokens_schedule == 'ada':
                assert assist_acc_head is not None, 'assist_acc_head is None!' 
                
                
                ### obtain current acceptance probability with assist_acc_head
                hidden_states = assistant_model_outputs.get("hidden_states")  # hidden_states[-1] is the last hidden states, size: bs * seq_len * hidden_dim
                logits = assist_acc_head(hidden_states[-1][0, -1].float())

                if stop_threshold is None:
                    logger.warning("[Deprecated] Stop_threshold not set. using the acceptance of current token instead.")
                    predicted = logits.argmax(dim = -1)
                    stop_prediction = (predicted == 0)
                else:

                    ## stop generation when the estimated P(exists one reject) = 1 - P(all proposed tokens are accepted) exceeds threshold.

                    if assist_steps == 1:
                        acc_prob = 1 # skip the first round as all tokens are verified and there are no proposed tokens.
                    else:
                        acc_prob = logits.softmax(dim = -1)[1].item()
                    cum_acc_prob *= acc_prob
                    rej_prob = 1 - cum_acc_prob
                    
                    stop_prediction = (rej_prob > stop_threshold)

                # bound = (min, max): forces the generated tokens to be inside [min, max] (both boundaries are included)
                if bound is not None:
                    if assist_steps >= bound[1]:
                        is_stop = True
                    elif assist_steps < bound[0]:
                        is_stop = False
                    else:
                        is_stop = stop_prediction
                else:
                    is_stop = stop_prediction
                        

                if is_stop:
                    break

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

        if candidate_length == 0:
            last_assistant_token_is_eos = False

        # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
        # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
        # we use this forward pass to also pick the subsequent logits in the original model.

        # 2.1. Run a forward pass on the candidate sequence
        if "past_key_values" in model_kwargs:
            model_attn = torch.ones_like(candidate_input_ids)
            model_input_ids = candidate_input_ids[:, -candidate_length - 1 :]
            if model.config.is_encoder_decoder:
                outputs = model(
                    decoder_input_ids=model_input_ids,
                    decoder_attention_mask=model_attn,
                    past_key_values=model_kwargs["past_key_values"],
                    encoder_outputs=model_kwargs["encoder_outputs"],
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    use_cache=True,
                )
            else:
                outputs = model(
                    model_input_ids,
                    attention_mask=model_attn,
                    past_key_values=model_kwargs["past_key_values"],
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    use_cache=True,
                )
        else:
            if model.config.is_encoder_decoder:
                outputs = model(
                    decoder_input_ids=candidate_input_ids,
                    encoder_outputs=model_kwargs["encoder_outputs"],
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    use_cache=True,
                )
            else:
                outputs = model(
                    candidate_input_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    use_cache=True,
                )

        # 2.2. Process the new logits
        new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
        if len(logits_processor) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
        if len(logits_warper) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

        # 3. Obtain the next tokens from the original model logits.
        if do_sample:
            # speculative decoding logit here.

            probs = new_logits[:, -candidate_length - 1 :, :].softmax(dim=-1)  # bs(1) * (candidate_length+1) * vocab_size
            p_prob, next_p = probs[:, :-1], probs[:, -1]

            if candidate_length == 0:
                n_matches = 0
            else:
                q_prob = torch.stack(q_prob, dim=1)   # bs(1) * candidate_length * vocab_size

                candidate_index = candidate_input_ids[:, -candidate_length:, None]

                q_candidate = q_prob.gather(-1, candidate_index).squeeze(-1)
                p_candidate = p_prob.gather(-1, candidate_index).squeeze(-1)
                r_candidate = torch.rand_like(q_candidate, device = q_candidate.device)
                n_matches = ((r_candidate > (p_candidate/q_candidate)).cumsum(dim=-1) < 1).sum()

        else:

            # greedy decoding logic.
            selected_tokens = new_logits[:, -candidate_length - 1 :, :].argmax(dim=-1)

            # 4. Compare the argmax from the original model logits with the assistant forecasted tokens. We can keep
            # the assistant forecasted tokens until the first mismatch, or until the max length is reached.
            if candidate_length == 0:
                n_matches = 0
            else:
                candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
                n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()






        # 5. Update variables according to the number of matching assistant tokens. Remember: the token generated
        # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
        # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
        # is no match.

        num_mismatched_tokens += (candidate_length - n_matches)


        if do_sample:
            # for speculative decoding
            candidate_new_tokens = candidate_input_ids[:, (candidate_input_ids.shape[1] - candidate_length):]
            valid_tokens = candidate_new_tokens[:, : n_matches]

            # for the last token
            ## case 1. rejected some token. resample from [p-q]+, index = n_matches
            if n_matches < candidate_length:
                next_p = p_prob[:, n_matches, :] - q_prob[:, n_matches, :] # bs(1) * vocab_size
                next_p.clamp_(min=0.)
                next_p = next_p / next_p.sum(dim = -1, keepdim=True)

            ## case 2. all tokens accepted: sample from next_p (defined before)

            new_added_token = torch.multinomial(next_p, num_samples=1)
            valid_tokens = torch.cat((valid_tokens, new_added_token), dim=-1)

        else:
            # for greedy decoding
            # 5.2. Get the valid continuation, after the matching tokens
            valid_tokens = selected_tokens[:, : n_matches + 1]

        # 5.1. Ensure we don't generate beyond max_len or an EOS token
        if last_assistant_token_is_eos and n_matches == candidate_length:
            n_matches -= 1
        n_matches = min(n_matches, max_len - cur_len - 1)
        valid_tokens = valid_tokens[:, : n_matches + 1]

        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        if streamer is not None:
            streamer.put(valid_tokens.cpu())
        new_cur_len = input_ids.shape[-1]


        # 5.3. Discard past key values relative to unused assistant tokens
        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(model, outputs.past_key_values, new_cache_size)
        if "assistant_past_key_values" in model_kwargs:
            model_kwargs["assistant_past_key_values"] = _crop_past_key_values(
                assistant_model, model_kwargs["assistant_past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

        # 6. Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.

        assist_rounds += 1
        if num_assistant_tokens_schedule == 'heuristic':
            if n_matches == int(assistant_model.max_assistant_tokens):
                assistant_model.max_assistant_tokens += 2.0
            else:
                assistant_model.max_assistant_tokens = max(1.0, assistant_model.max_assistant_tokens - 1.0)
        elif num_assistant_tokens_schedule == 'oracle':
            if assist_rounds < len(oracle_token_num_list):
                assistant_model.max_assistant_tokens = oracle_token_num_list[assist_rounds]
            else:
                logger.warning("warning. assist_rounds exceed len(oracle_token_num_list)")
            # print("oracle token num: %d" % assistant_model.max_assistant_tokens)


        # Assistant: main logic end

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Store scores, attentions and hidden_states when required
        # Assistant: modified to append one tuple element per token, as in the other generation methods.
        if return_dict_in_generate:
            if output_scores:
                scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))

            if "past_key_values" not in model_kwargs:
                added_len = new_cur_len
            else:
                added_len = n_matches + 1

            if output_attentions:
                if model.config.is_encoder_decoder:
                    cross_attentions = _split_model_outputs(
                        cross_attentions, outputs.cross_attentions, cur_len, added_len
                    )
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.decoder_attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
                else:
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
            if output_hidden_states:
                if model.config.is_encoder_decoder:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                    )
                else:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                    )

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                input_ids[:, -1]
                .tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if model.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids, num_mismatched_tokens.item(), assist_rounds

