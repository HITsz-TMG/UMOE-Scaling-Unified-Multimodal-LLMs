# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
import json
import logging
import pathlib
from pprint import pprint
from typing import Dict, Optional, Sequence, List, Mapping, Any
import datasets
import torch
import transformers
import tokenizers
import re
import deepspeed
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import TrainerCallback, AutoProcessor
from moe_trainer import MoETrainer
from transformers.integrations import WandbCallback

from training_utils import rank0_print, rank0_pprint, MYEpochSaveCallback, set_trainable, compress_strings_set
from Models.UniMoEV2 import UniMoEV2Qwen2VLForConditionalGeneration, UniMoEV2Qwen2VLConfig
from transformers import Qwen2VLForConditionalGeneration
from DataLoaders.qwen2vl_datasets import LazySupervisedDataset, DataCollatorForSupervisedDataset
import DataLoaders.qwen2vl_datasets

DataLoaders.qwen2vl_datasets.MAX_PIXELS = 512 * 28 * 28


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    initialize: bool = field(default=False)
    # moe
    mlp_dynamic_expert_num: int = field(default=4)
    mlp_dynamic_top_p: float = field(default=0.7)
    mlp_dynamic_top_k: float = field(default=2)
    ignore_differentiable_router: bool = field(default=False)
    mlp_fixed_expert_num: int = field(default=2)
    mlp_dynamic_null_expert_num: int = field(default=1)
    token_drop: bool = field(default=False)
    drop_policy: str = field(default="probs") 
    min_capacity: int = field(default=8)
    min_capacity: int = field(default=8)
    capacity_factor: float = field(default=1.0)
    fp32_gate: bool = field(default=True)
    ep_size: int = field(default=1)
    fixed_ep_size: int = field(default=1)
    dynamic_mlp_size_factor: int = field(default=1)
    fixed_mlp_size_factor: int = field(default=1)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    processor_path: str = field(default=None)
    image_root: str = field(default=None)
    aux_balance_weight: float = field(default=None)
    data_sample: int = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    attn_implementation: str = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    l_aux_weight: float = field(default=0.025)
    min_l_aux_weight: float = field(default=0.001)
    l_aux_weight_decay_steps: int = field(default=10000)
    moe_copy: str = field(default=None)
    drop_token_num_print: bool = field(default=True) 
    only_gate_training: bool = field(default=False)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, image_processor) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        image_processor=image_processor,
        image_root=data_args.image_root,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, aux_balance_weight=data_args.aux_balance_weight)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def initial_model(model_path, training_args, model_args):
    rank0_print("________ initial start ________")
    model_dtype = torch.bfloat16 if training_args.bf16 else None

    rank0_print("________ initial model ________")
    qwen2vl_model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, cache_dir=training_args.cache_dir, torch_dtype=model_dtype, attn_implementation=training_args.attn_implementation)

    rank0_print("________ initial config ________")

    qwen2vl_config = qwen2vl_model.config.to_dict()

    new_qwen2vl_config = {
        "mlp_dynamic_expert_num": model_args.mlp_dynamic_expert_num,
        "mlp_dynamic_top_p": model_args.mlp_dynamic_top_p,
        "mlp_dynamic_top_k": model_args.mlp_dynamic_top_k,
        "mlp_dynamic_null_expert_num": model_args.mlp_dynamic_null_expert_num,
        "mlp_fixed_expert_num": model_args.mlp_fixed_expert_num,
        "drop_token_num_print": training_args.drop_token_num_print,
        "dynamic_intermediate_size": qwen2vl_config["intermediate_size"] // model_args.dynamic_mlp_size_factor,
        "shared_intermediate_size": qwen2vl_config["intermediate_size"] // model_args.fixed_mlp_size_factor,
        "ep_size": model_args.ep_size,
        "fixed_ep_size": model_args.fixed_ep_size,
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.01,
        "token_drop": model_args.token_drop,
        "drop_policy": model_args.drop_policy,
        "min_capacity": model_args.min_capacity,
        "capacity_factor": model_args.capacity_factor,
        "fp32_gate": model_args.fp32_gate,
        "l_aux_weight": training_args.l_aux_weight,  #  0.025
    }
    qwen2vl_config = UniMoEV2Qwen2VLConfig.from_dict(
        {
            **qwen2vl_config,
            **new_qwen2vl_config,
        }
    )

    model = UniMoEV2Qwen2VLForConditionalGeneration._from_config(qwen2vl_config, attn_implementation=training_args.attn_implementation, torch_dtype=model_dtype)

    rank0_print("________ load model ________")
  
    state_dict = qwen2vl_model.state_dict()

    # MoE 加载策略,
    # moe_copy=all 表示所有dynamic expert都是复制dense, shared复制矩阵剪切的
    # moe_copy=single 表示只有一个expert复制dense
    rank0_print(f"[INFO] MoE initial strategy: {training_args.moe_copy}")
    mlp_state_dict = {}
    mlp_pattern = r"model\.layers\.(\d+)\.mlp"
    for k in list(state_dict.keys()):
        match = re.match(mlp_pattern, k)
        if match:
            mlp_state_dict[k] = state_dict.pop(k)

    not_load_but_init_weight = set()
    cutted_weight = set()
    cutted_offset = {}
    cutted_full_cnt = {}
    cur_model_state_dict = model.state_dict()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ep_group_rank = local_rank % model_args.ep_size

    if training_args.moe_copy != "none":
        mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(fixed_real_moe|dynamic_real_moe\.deepspeed_moe\.experts\.deepspeed_experts)\.(\d+)"
        for n in list(cur_model_state_dict.keys()):
            match = re.match(mlp_pattern, n)
            if match:
                layer = int(match.group(1))
                expert_type = match.group(2)
                expert = int(match.group(3))
                rest = n[len(match.group(0)) :]
                source_mlp_name = f"model.layers.{layer}.mlp{rest}"
                source_mlp = mlp_state_dict[source_mlp_name]
                if training_args.moe_copy == "all" or expert == 0:
                    cur_mlp = cur_model_state_dict[n]  

                    if cur_mlp.shape[0] != source_mlp.shape[0]:
                        if expert_type != "fixed_real_moe":
                            if source_mlp_name not in cutted_offset:
                                initial_offset = (ep_group_rank * (model_args.mlp_dynamic_expert_num // model_args.ep_size) * cur_mlp.shape[0]) % source_mlp.shape[0]
                                initial_souce_id = (ep_group_rank * (model_args.mlp_dynamic_expert_num // model_args.ep_size) * cur_mlp.shape[0]) / source_mlp.shape[0]

                                cutted_offset[source_mlp_name] = initial_offset
                                cutted_full_cnt[source_mlp_name] = 0
                            offset = cutted_offset[source_mlp_name]
                        else:
                            offset = 0

                        state_dict[n] = source_mlp[offset : offset + cur_mlp.shape[0]] 

                        if expert_type != "fixed_real_moe":
                            cutted_offset[source_mlp_name] += cur_mlp.shape[0]
                            if cutted_offset[source_mlp_name] >= source_mlp.shape[0]:
                                assert cutted_offset[source_mlp_name] == source_mlp.shape[0], f"{source_mlp.shape[0]} {cur_mlp.shape[0]}, {cutted_offset[source_mlp_name]}"
                                cutted_offset[source_mlp_name] %= source_mlp.shape[0]
                            cutted_full_cnt[source_mlp_name] += cur_mlp.shape[0] / source_mlp.shape[0]

                        cutted_weight.add(n)
                    elif cur_mlp.shape[1] != source_mlp.shape[1]:
                        if expert_type != "fixed_real_moe":
                            if source_mlp_name not in cutted_offset:
                                initial_offset = (ep_group_rank * (model_args.mlp_dynamic_expert_num // model_args.ep_size) * cur_mlp.shape[1]) % source_mlp.shape[1]
                                initial_souce_id = (ep_group_rank * (model_args.mlp_dynamic_expert_num // model_args.ep_size) * cur_mlp.shape[1]) / source_mlp.shape[1]

                                cutted_offset[source_mlp_name] = 0
                                cutted_full_cnt[source_mlp_name] = 0
                            offset = cutted_offset[source_mlp_name]
                        else:
                            offset = 0

                        state_dict[n] = source_mlp[:, offset : offset + cur_mlp.shape[1]] 

                        if expert_type != "fixed_real_moe":
                            cutted_offset[source_mlp_name] += cur_mlp.shape[1]
                            if cutted_offset[source_mlp_name] >= source_mlp.shape[1]:
                                assert cutted_offset[source_mlp_name] == source_mlp.shape[1], f"{source_mlp.shape[1]} {cur_mlp.shape[1]}, {cutted_offset[source_mlp_name]}"
                                cutted_offset[source_mlp_name] %= source_mlp.shape[1]
                            cutted_full_cnt[source_mlp_name] += cur_mlp.shape[1] / source_mlp.shape[1]

                        cutted_weight.add(n)
                    else:
                        state_dict[n] = source_mlp
                elif training_args.moe_copy == "single":
                    not_load_but_init_weight.add(n)
                    if n.endswith("weight"):
                        state_dict[n] = torch.empty_like(source_mlp)
                        state_dict[n].normal_(mean=0.0, std=qwen2vl_config.initializer_range)
                    else:
                        assert n.endswith("bias")
                        state_dict[n] = torch.zeros_like(source_mlp)

    s1, s2, s3, s4, s5 = state_dict["visual.patch_embed.proj.weight"].shape
    state_dict["visual.patch_embed.proj.weight"] = state_dict["visual.patch_embed.proj.weight"].view(s1, -1, s4, s5)

    not_loaded_weight = set(model.state_dict().keys()) - set(state_dict.keys())
    omit_weight = set(state_dict.keys()) - set(model.state_dict().keys())
    assert len(omit_weight) == 0, f"omit_weight: {omit_weight}"
    rank0_print(f"not_loaded_weight: ", compress_strings_set(not_loaded_weight))
    rank0_print(f"cutted_loaded_weight: ", compress_strings_set(cutted_weight))
    rank0_print(f"not_loaded_but_init_weight: ", compress_strings_set(not_load_but_init_weight))
    rank0_pprint(cutted_full_cnt)

    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        assert param.dtype == model_dtype, f"{name} dtype is {param.dtype} but not {model_dtype}"

    rank0_print("________ initial done ________")

    return model


def post_process_for_moe(model):
    for l in range(len(model.base_model.model.language_model.model.layers)):
        for expert in model.base_model.model.language_model.model.layers[l].mlp.dynamic_real_moe.deepspeed_moe.experts.deepspeed_experts:
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = model.base_model.model.language_model.model.layers[l].mlp.expert_group_name
    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("model_args")
    rank0_print(model_args.__dict__)
    rank0_print("data_args")
    rank0_print(data_args.__dict__)
    rank0_print("training_args")
    rank0_print(training_args.__dict__)

    if model_args.initialize:
        model = initial_model(
            model_path=model_args.model_name_or_path,
            training_args=training_args,
            model_args=model_args,
        )
    else:
        model = UniMoEV2Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
        )
    model.config.use_cache = False
    rank0_print("l_aux_weight: ", model.config.l_aux_weight)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    image_processor = AutoProcessor.from_pretrained(data_args.processor_path).image_processor
    processor = AutoProcessor.from_pretrained(data_args.processor_path)

    training_module_pattern = None
    if training_args.only_gate_training:
        training_module_pattern = [r"model\.layers\.(\d+)\.mlp\.gate"]
        print(f"[Code] only_gate_training is setting to True !")
    set_trainable(model, training_module_pattern=training_module_pattern, log=True) 

    if training_args.gradient_checkpointing:
        rank0_print(f"[Code] if training_args.gradient_checkpointing")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, image_processor=image_processor)

    trainer = MoETrainer(
        model=model, tokenizer=tokenizer, args=training_args, callbacks=[MYEpochSaveCallback(save_processor=processor, save_dir=training_args.output_dir), WandbCallback()], **data_module
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()


if __name__ == "__main__":
    train()
