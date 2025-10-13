import copy
import json
import logging
import os
import pathlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, List, Mapping, Optional, Sequence
import datasets
import deepspeed
import tokenizers
import torch
import transformers
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)
from transformers import AutoProcessor, TrainerCallback, TrainingArguments

def rank0_pprint(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            pprint(*args)
    else:
        pprint(*args)


def rank0_print(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class MYEpochSaveCallback(TrainerCallback):
    def __init__(self, save_model=None, save_dir=None, save_processor=None, save_lora_base_model=False):
        self.save_model = save_model
        self.save_dir = save_dir
        self.save_processor = save_processor
        self.save_lora_base_model = save_lora_base_model

    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        # Save
        control.should_save = True
        self.__save_model__(args, state, control, prefix="epoch", **kwargs)
        return control

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        self.__save_model__(args, state, control, prefix="checkpoint", **kwargs)
        return control

    def __save_model__(self, args: TrainingArguments, state, control, prefix="checkpoint", **kwargs):
        if self.save_dir is not None and torch.distributed.get_rank() == 0:
            save_dir = os.path.join(self.save_dir, f"{prefix}-{state.global_step}")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            if self.save_model is not None:
                self.save_model.save_pretrained(save_dir)
                if self.save_lora_base_model:
                    self.save_model.base_model.save_pretrained(save_dir)

            if self.save_processor is not None:
                self.save_processor.save_pretrained(save_dir)


def set_trainable(model, training_module_pattern=None, log=True):
    if training_module_pattern is None:
        model.requires_grad_(True)
    else:
        if isinstance(training_module_pattern, str):
            training_module_pattern = [training_module_pattern]
        assert isinstance(training_module_pattern, List)

        model.requires_grad_(False)
        for n, m in model.named_modules():
            if any([re.match(p, n) for p in training_module_pattern]):
                m.requires_grad_(True)

    if log:
        all_param = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            if param.requires_grad:
                trainable_params += num_params
                rank0_print(name, num_params)
            all_param += num_params

        rank0_print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    return model


def get_peft_config(model_args, training_args):
    def get_attr(self, att, default=None):
        return getattr(self, att) if hasattr(self, att) else default

    peft_mode = model_args.peft_mode
    if peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=get_attr(model_args, "lora_target_modules", ["q_proj", "v_proj"]),
            r=get_attr(model_args, "lora_r", 16),
            lora_alpha=get_attr(model_args, "lora_alpha", 32),
            lora_dropout=get_attr(training_args, "lora_dropout", 0.05),
        )
    elif peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=10,
            encoder_hidden_size=512,
            prefix_projection=True,
        )
    elif peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=10,
            encoder_hidden_size=512,
        )
    elif peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=10,
        )
    else:
        raise KeyError(peft_mode)
    return peft_config


def prepare_peft_model(model, model_args, training_args, log=True):
    config = get_peft_config(model_args, training_args)
    model = get_peft_model(model, config)
    if log:
        model.print_trainable_parameters()
    return model


def prepare_model_for_gradient_checkpointing(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


def compress_strings_set(strings):
    holder_str = "<number_holder>"

    def split_and_classify(s):
        parts = s.split(".")
        value = None
        key_parts = []
        find_digital = False
        for part in parts:
            if part.isdigit() and not find_digital:
                find_digital = True
                value = int(part)
                key_parts.append(holder_str)
            else:
                key_parts.append(part)
        key = ".".join(key_parts)
        return value, key

    def compress_numeric_parts(numeric_parts):
        numeric_parts.sort()
        ranges = []

        if not numeric_parts:
            return numeric_parts
        start = end = numeric_parts[0]
        for num in numeric_parts[1:]:
            if num == end + 1:
                end = num
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = num
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        return ranges

    while True:
        grouped = {}
        for s in strings:
            value, key = split_and_classify(s)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(value)

        result = []
        for key, values in grouped.items():
            numeric_ranges = compress_numeric_parts(values)
            if numeric_ranges:
                numeric_str = f"[{','.join(numeric_ranges)}]"
            else:
                numeric_str = ""
            result.append(key.replace(holder_str, numeric_str))

        if len(result) == len(strings):
            break
        else:
            strings = result

    return set(result)
