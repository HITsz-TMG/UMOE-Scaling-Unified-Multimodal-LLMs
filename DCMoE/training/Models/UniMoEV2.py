# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2-VL model."""

import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from deepspeed import comm as dist
from deepspeed.moe.mappings import drop_tokens, gather_tokens
from deepspeed.moe.sharded_moe import FIRST_ALLTOALL_TIMER, MOE_TIMER, SECOND_ALLTOALL_TIMER, _AllToAll, einsum, gumbel_rsample
from deepspeed.utils import groups, log_dist
from deepspeed.utils.timer import SynchronizedWallClockTimer
from torch import Tensor
from torch.func import functional_call, vmap
from torch.nn import CrossEntropyLoss, LayerNorm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    QWEN2_VL_ATTENTION_CLASSES,
    # Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2VLRotaryEmbedding,
    _prepare_4d_causal_attention_mask_with_cache_position,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from .MoE_utils import compress_matrix, decompress_matrix
from .npu_qwen2vl import Qwen2VisionTransformerPretrainedModel

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None

logger = logging.get_logger(__name__)

FAST_INIT = True
if FAST_INIT:
    logger.warning(f"using FAST initial for UniMoEV2 Qwen2_vl !!!")

class UniMoEV2Qwen2VLConfig(Qwen2VLConfig):
    model_type = "UniMoEV2_qwen2_vl"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        mlp_dynamic_expert_num=4,
        mlp_dynamic_null_expert_num=0,
        mlp_dynamic_top_p=0.7,
        mlp_dynamic_top_k=2,
        mlp_fixed_expert_num=2,
        dynamic_intermediate_size=8960,
        shared_intermediate_size=8960,
        enable_expert_tensor_parallelism: bool = False,
        ep_size=1,                  # deepspeed moe ep
        fixed_ep_size=1,
        router_jitter_noise=0.01,
        input_jitter_noise=0.01,
        token_drop=False,
        drop_policy: str = "probs",  # probs, position
        min_capacity: int = 8,
        capacity_factor: float = 1.0,
        fp32_gate=True,
        avg_hidden_states_last=False,
        drop_token_num_print=True,
        l_aux_weight=0,
        **kwargs,
    ):
        self.mlp_dynamic_expert_num = mlp_dynamic_expert_num
        self.mlp_dynamic_top_p = mlp_dynamic_top_p
        self.mlp_dynamic_top_k = mlp_dynamic_top_k
        self.mlp_fixed_expert_num = mlp_fixed_expert_num
        self.mlp_dynamic_null_expert_num = mlp_dynamic_null_expert_num

        self.dynamic_intermediate_size = dynamic_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size

        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        self.ep_size = ep_size
        self.fixed_ep_size = fixed_ep_size

        self.input_jitter_noise = input_jitter_noise
        self.router_jitter_noise = router_jitter_noise

        self.token_drop = token_drop
        self.drop_policy = drop_policy
        self.min_capacity = min_capacity
        self.capacity_factor = capacity_factor

        self.fp32_gate = fp32_gate
        self.avg_hidden_states_last = avg_hidden_states_last
        self.drop_token_num_print = drop_token_num_print

        self.l_aux_weight = l_aux_weight

        super().__init__(**kwargs)


@dataclass
class MoEQwen2VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    all_router_logits: Tuple = None
    all_router_top_k: Tuple = None
    all_router_expert_mask: Tuple = None
    all_router_weight: Tuple = None
    aux_balance_loss: torch.FloatTensor = None


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_router_logits: Tuple = None
    all_router_top_k: Tuple = None
    all_router_weight: Tuple = None
    all_router_expert_mask: Tuple = None
    all_aux_loss: Tuple = None


class SharedExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.shared_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class DynamicExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.dynamic_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class NULLExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_state):
        # return hidden_state * 0
        return torch.zeros_like(hidden_state, dtype=hidden_state.dtype, device=hidden_state.device)

class mp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        multiplier: torch.Tensor,
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
        ctx,
        grad_at_output: torch.Tensor,
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (grad_at_scores_expaned,None,None,None,None)


def allocation_expert(scores, top_k, jitter_eps):
    masked_scores = scores
    multiplier_list = []
    selected_experts_list = []

    for _ in range(top_k):
        with torch.no_grad():
            mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
            factor = scores.abs().clamp(min=mask_logits_threshold.abs())  
            mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)  


        masked_gates = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))
        selected_experts = max_ind
        masked_gates = torch.softmax(masked_gates, dim=-1)
        multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
        multiplier = multiplier_o

        masked_scores = torch.scatter(
            masked_scores,
            -1,
            selected_experts,
            float("-inf"),
        )

        multiplier_list.append(multiplier)
        selected_experts_list.append(selected_experts)

    multiplier = torch.concat(multiplier_list, dim=-1)
    selected_experts = torch.concat(selected_experts_list, dim=-1)

    return (
        multiplier,
        selected_experts,
    )


def dynamic_expert_selection(logits, top_p):
    dynamic_scores = torch.softmax(logits, dim=-1)
    dynamic_scores_sorted, _ = torch.sort(dynamic_scores, dim=-1, descending=True)
    dynamic_scores_cumsum = dynamic_scores_sorted.cumsum(dim=-1)
    dynamic_top_k = (~(dynamic_scores_cumsum >= top_p)).sum(dim=-1)
    dynamic_top_k = dynamic_top_k + 1
    return dynamic_top_k


def _capacity(num_tokens, num_experts, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


def cal_global_weight(
    expert_mask: torch.Tensor,
    full_router_logits: torch.Tensor,
    mlp_dynamic_expert_num: int,
    routing_weights: torch.Tensor,
):
    global_weight = torch.softmax(full_router_logits.masked_fill(expert_mask == 0, float("-inf")), dim=-1)
    global_dynamic_weight = global_weight[:, :mlp_dynamic_expert_num]
    global_fixed_weight = global_weight[:, mlp_dynamic_expert_num:]
    global_dynamic_weight = routing_weights * global_dynamic_weight.sum(-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1])  # 计算dynamic的weight缩放因数
    global_weight = torch.cat((global_dynamic_weight, global_fixed_weight), dim=-1)
    return global_weight


class UniMoEV2MoESparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.mlp_dynamic_expert_num = config.mlp_dynamic_expert_num + config.mlp_dynamic_null_expert_num
        self.mlp_dynamic_real_expert_num = config.mlp_dynamic_expert_num
        self.mlp_dynamic_null_expert_num = config.mlp_dynamic_null_expert_num
        self.mlp_dynamic_top_p = config.mlp_dynamic_top_p
        self.mlp_dynamic_top_k = config.mlp_dynamic_top_k
        self.mlp_fixed_expert_num = config.mlp_fixed_expert_num
        self.num_experts = self.mlp_dynamic_expert_num + self.mlp_fixed_expert_num

        if self.mlp_dynamic_top_p == 0:
            print(f"mlp_dynamic_top_p is 0, will use mlp_dynamic_top_k={self.mlp_dynamic_top_k} instead !!!")

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.fixed_real_moe = nn.ModuleList([SharedExpertMLP(config) for _ in range(self.mlp_fixed_expert_num)])
        self.dynamic_real_moe = MoE(config, DynamicExpertMLP(config), self.mlp_dynamic_real_expert_num, config.ep_size)
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise
        self.min_capacity = config.min_capacity
        self.capacity_factor = config.capacity_factor
        self.token_drop = config.token_drop
        self.drop_policy = config.drop_policy
        self.avg_hidden_states_last = config.avg_hidden_states_last
        self.drop_token_num_print = config.drop_token_num_print
        self.fp32_gate = config.fp32_gate

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, aux_balance_weight: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        original_hidden_states = hidden_states

        if self.training and self.fp32_gate:
            hidden_states = hidden_states.float()

        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise)

        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.training and self.fp32_gate:
            full_router_logits = torch.nn.functional.linear(hidden_states, weight=self.gate.weight.float(), bias=None)
        else:
            full_router_logits = self.gate(hidden_states)
        dynamic_router_logits = full_router_logits[:, : self.mlp_dynamic_expert_num]

        if self.mlp_dynamic_top_p != 0:
            dynamic_top_k = dynamic_expert_selection(dynamic_router_logits, self.mlp_dynamic_top_p)
        else:
            dynamic_top_k = torch.full((dynamic_router_logits.shape[0],), self.mlp_dynamic_top_k, dtype=torch.int, device=dynamic_router_logits.device)

        expert_mask = torch.zeros((batch_size * sequence_length, self.num_experts), dtype=torch.int, device=hidden_states.device)
        routing_weights = torch.zeros((batch_size * sequence_length, self.mlp_dynamic_expert_num), dtype=hidden_states.dtype, device=hidden_states.device)
        
        for top_k in range(1, self.mlp_dynamic_expert_num + 1):
            group_idx = torch.nonzero(dynamic_top_k == top_k, as_tuple=True)[0]
            if len(group_idx) == 0:
                continue

            dynamic_group_logits = dynamic_router_logits[group_idx]
            group_routing_weights, group_selected_experts = allocation_expert(
                dynamic_group_logits,
                top_k=top_k,
                jitter_eps=self.router_jitter_noise,
            )
            group_expert_mask = torch.nn.functional.one_hot(group_selected_experts, num_classes=self.num_experts)
            group_expert_mask = group_expert_mask.sum(dim=1)

            group_weight = torch.zeros((len(group_idx), self.mlp_dynamic_expert_num), dtype=hidden_states.dtype, device=hidden_states.device)
            group_weight.scatter_(dim=-1, index=group_selected_experts, src=group_routing_weights)
            routing_weights.index_add_(0, group_idx, group_weight)
            expert_mask.index_add_(0, group_idx, group_expert_mask.to(expert_mask.dtype))

        routing_weights = routing_weights / (routing_weights.sum(dim=-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1]) + 1e-6)


        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, f"{attention_mask.shape}" 
            attention_mask = attention_mask.to(expert_mask.dtype).view(-1).unsqueeze(-1).expand(-1, self.num_experts)
            expert_mask = expert_mask * attention_mask

        if self.mlp_dynamic_expert_num < self.num_experts:
            expert_mask[:, self.mlp_dynamic_expert_num :] = 1  

        aux_loss = load_balancing_loss_func(
            expert_mask=expert_mask,
            mlp_dynamic_expert_num=self.mlp_dynamic_expert_num,
            global_weight=None,
            full_router_logits=full_router_logits,
            routing_weights=routing_weights,
            aux_balance_weight=aux_balance_weight,
        )

        if self.token_drop:
            expert_mask_dtype = expert_mask.dtype
            capacity = _capacity(batch_size * sequence_length, self.mlp_dynamic_expert_num, torch.tensor(self.capacity_factor), torch.tensor(self.min_capacity))
            if self.drop_policy == "probs":
                if capacity > dynamic_router_logits.shape[0]:
                    print(f"[warning] token capacity({capacity}) > token num({dynamic_router_logits.shape[0]}), setting capacity=token num")
                    capacity = dynamic_router_logits.shape[0]
                dynamic_expert_mask = expert_mask[:, : self.mlp_dynamic_expert_num].bool()
                token_drop_router_logits = torch.masked_fill(dynamic_router_logits, ~dynamic_expert_mask, torch.finfo(dynamic_router_logits.dtype).min)
                capacity_probs, capacity_indices = torch.topk(token_drop_router_logits, k=capacity, dim=0, sorted=False)
                capacity_mask = torch.zeros_like(expert_mask).scatter(0, capacity_indices, 1)
                capacity_mask[:, self.mlp_dynamic_expert_num :] = 1
                expert_mask = torch.logical_and(expert_mask, capacity_mask)

                ori_token_num = dynamic_expert_mask.sum().item()
                cur_token_num = expert_mask[:, : self.mlp_dynamic_expert_num].sum().item()
                if self.drop_token_num_print and ("RANK" not in os.environ or int(os.environ["RANK"]) == 0):
                    print(f"drop {ori_token_num - cur_token_num} tokens from total {ori_token_num} tokens")

            elif self.drop_policy == "position":
                locations = torch.cumsum(expert_mask, dim=0) - 1
                expert_mask *= torch.lt(locations, capacity)
            else:
                raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
            expert_mask = expert_mask.to(expert_mask_dtype)

            routing_weights = routing_weights.masked_fill(~(expert_mask[:, : self.mlp_dynamic_expert_num].bool()), 0.0)
            routing_weights = routing_weights / (routing_weights.sum(dim=-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1]) + 1e-6)

        if self.mlp_dynamic_expert_num < self.num_experts:
            global_weight = cal_global_weight(expert_mask, full_router_logits, self.mlp_dynamic_expert_num, routing_weights)
        else:
            global_weight = routing_weights
        assert ((expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)).sum() == 0

        hidden_states = original_hidden_states.view(-1, hidden_dim)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        global_weight = global_weight.to(hidden_states.dtype)
        current_hidden_states = self.dynamic_real_moe(hidden_states, expert_mask=expert_mask[:, : self.mlp_dynamic_real_expert_num], router_weight=global_weight[:, : self.mlp_dynamic_real_expert_num])
        final_hidden_states = final_hidden_states + current_hidden_states

        for expert_idx in range(self.mlp_fixed_expert_num):
            expert_layer = self.fixed_real_moe[expert_idx]
            current_state = hidden_states
            current_global_weight = global_weight[:, self.mlp_dynamic_expert_num + expert_idx].unsqueeze(-1) 
            current_hidden_states = expert_layer(current_state) * current_global_weight
            final_hidden_states = final_hidden_states + current_hidden_states

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if not self.training and self.avg_hidden_states_last:
            dist.all_reduce(final_hidden_states, op=dist.ReduceOp.AVG, group=self.dynamic_real_moe.deepspeed_moe.ep_group)

        return final_hidden_states, full_router_logits, dynamic_top_k, expert_mask, global_weight, aux_loss


def load_balancing_loss_func(
    expert_mask: torch.Tensor,
    mlp_dynamic_expert_num: int,
    global_weight: Optional[torch.Tensor] = None,
    full_router_logits: Optional[torch.Tensor] = None,
    routing_weights: Optional[torch.Tensor] = None,
    aux_balance_weight: Optional[torch.Tensor] = None,
    version=2,
) -> float:
    min_dtype = torch.finfo(full_router_logits.dtype).min 
    global_weight = full_router_logits.masked_fill(expert_mask == 0, min_dtype)
    global_weight = global_weight[:, :mlp_dynamic_expert_num]
    global_weight = torch.softmax(global_weight, dim=-1)
    expert_mask = expert_mask[:, :mlp_dynamic_expert_num]
    num_experts = expert_mask.shape[-1]
    if aux_balance_weight is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(global_weight, dim=0)
    else:
        batch_size, sequence_length = aux_balance_weight.shape
        num_hidden_layers = global_weight.shape[0] // (batch_size * sequence_length)
        expert_attention_mask = aux_balance_weight[None, :, :, None].expand((num_hidden_layers, batch_size, sequence_length, num_experts)).reshape(-1, num_experts).to(global_weight.device)
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
        router_prob_per_expert = torch.sum(global_weight * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)

    return overall_loss * num_experts

class Experts(deepspeed.moe.experts.Experts):
    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(deepspeed.moe.experts.Experts, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        for expert in self.deepspeed_experts:
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output


class MOELayer(deepspeed.moe.sharded_moe.MOELayer):
    def __init__(
        self,
        experts: nn.Module,
        ep_group_name,
        ep_size,
        num_local_experts: int,
    ) -> None:
        super(deepspeed.moe.sharded_moe.MOELayer, self).__init__()

        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

    # copy
    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, hidden_states: Tensor, expert_mask: Tensor, router_weight: Tensor) -> Tensor:
        router_weight = router_weight * expert_mask

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).start()

        d_model = hidden_states.shape[-1]
        seq_len = hidden_states.shape[0]
        expert_num = expert_mask.shape[-1]
        capacity = expert_mask.sum(dim=0).max()
        if self.ep_group is not None:
            dist.all_reduce(capacity, op=dist.ReduceOp.MAX, group=self.ep_group)

        compres_hidden_states = hidden_states.unsqueeze(1).expand(seq_len, expert_num, d_model)  
        compres_hidden_states = compress_matrix(compres_hidden_states, expert_mask, force_dim=capacity, allow_larger_dim=True)  
        compres_expert_mask = compress_matrix(expert_mask, expert_mask, force_dim=capacity, allow_larger_dim=True)
        dispatched_input = einsum("ce,cem->ecm", compres_expert_mask, compres_hidden_states)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).start()

        assert deepspeed.utils.groups.mpu is None
        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)

        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        expert_output = self.experts(dispatched_input)

        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).start()

        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)

        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        expert_output = decompress_matrix(expert_output.transpose(0, 1), expert_mask, allow_larger_dim=True)
        combined_output = einsum("se,sem->sm", router_weight, expert_output)

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).stop()
            self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        return combined_output


class MoE(deepspeed.moe.layer.MoE):
    def __init__(self, config, expert, num_experts, ep_size, moe_name_prefix="ep_size"):
        super(deepspeed.moe.layer.MoE, self).__init__()

        self.enable_expert_tensor_parallelism = config.enable_expert_tensor_parallelism
        self.ep_size = ep_size
        self.num_experts = num_experts

        assert self.num_experts % self.ep_size == 0, f"Number of experts ({self.num_experts}) should be divisible by expert parallel size ({self.ep_size})"

        self.expert_group_name = f"{moe_name_prefix}_{self.ep_size}"
        self.num_local_experts = self.num_experts // self.ep_size

        log_dist(f"Creating MoE layer with num_experts: {self.num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}", [0])

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = MOELayer(experts, self.expert_group_name, self.ep_size, self.num_local_experts)

    def set_deepspeed_parallelism(self, use_data_before_expert_parallel_=False):
        self._create_process_groups(use_data_before_expert_parallel_=use_data_before_expert_parallel_)

    def _create_process_groups(self, use_data_before_expert_parallel_=False):
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                groups._create_expert_and_data_parallel(self.ep_size, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            else:
                groups._create_expert_data_and_model_parallel(self.ep_size, mpu=groups.mpu, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self, *input_args, **input_kwargs):
        return self.deepspeed_moe(*input_args, **input_kwargs)

class UniMoEV2Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; unexpected results may be encountered.")
        self.self_attn = QWEN2_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = UniMoEV2MoESparseMoeBlock(config)

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aux_balance_weight: Optional[torch.Tensor] = None,
        padding_token_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits_and_topk: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits, router_top_k, router_expert_mask, router_weight, aux_loss = self.mlp(hidden_states, padding_token_mask, aux_balance_weight)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # MoE Flag - return
        if output_router_logits_and_topk:
            outputs += (router_logits,)
            outputs += (router_top_k,)
        outputs += (router_expert_mask,)
        outputs += (router_weight,)
        outputs += (aux_loss,)

        return outputs


class UniMoEV2Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = UniMoEV2Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UniMoEV2Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if FAST_INIT:
            """
            only initialize the gate linear since we will load state dict for other parameters
            """
            if isinstance(module, UniMoEV2MoESparseMoeBlock):
                module.gate.weight.data.normal_(mean=0.0, std=std)
                if module.gate.bias is not None:
                    module.gate.bias.data.zero_()
        else:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()


class UniMoEV2Qwen2VLModel(UniMoEV2Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([UniMoEV2Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        aux_balance_weight: Optional[torch.Tensor] = None,
        padding_token_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits_and_topk: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits_and_topk else None
        all_router_top_k = () if output_router_logits_and_topk else None
        all_router_expert_mask = ()
        all_router_weight = ()
        all_aux_loss = ()
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    aux_balance_weight,
                    padding_token_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits_and_topk,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    aux_balance_weight=aux_balance_weight,
                    padding_token_mask=padding_token_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits_and_topk=output_router_logits_and_topk,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits_and_topk:
                all_router_logits += (layer_outputs[-5],)
                all_router_top_k += (layer_outputs[-4],)
            all_router_expert_mask += (layer_outputs[-3],)
            all_router_weight += (layer_outputs[-2],)
            all_aux_loss += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits, all_router_top_k, all_router_expert_mask, all_router_weight, all_aux_loss] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            all_router_logits=all_router_logits,
            all_router_top_k=all_router_top_k,
            all_router_expert_mask=all_router_expert_mask,
            all_aux_loss=all_aux_loss,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if self.config._attn_implementation == "sdpa" and attention_mask is not None and attention_mask.device.type == "cuda" and not output_attentions:
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class UniMoEV2Qwen2VLForConditionalGeneration(UniMoEV2Qwen2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config, attn_implementation=config._attn_implementation)
        self.model = UniMoEV2Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  
        self.l_aux_weight = config.l_aux_weight
        self.mlp_dynamic_expert_num = config.mlp_dynamic_expert_num + config.mlp_dynamic_null_expert_num
        self.mlp_fixed_expert_num = config.mlp_fixed_expert_num
        self.input_max_length = 0
        self.training_steps = 0

        self.post_init()

    @property
    def cur_aux_weight(self):
        return self.l_aux_weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        aux_balance_weight: Optional[torch.LongTensor] = None,
        padding_token_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits_and_topk: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoEQwen2VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if aux_balance_weight is not None:
            aux_balance_weight = attention_mask * aux_balance_weight

        if padding_token_mask is None:
            assert len(attention_mask.shape) == 2, f"{attention_mask.shape}" 
            padding_token_mask = attention_mask.bool()

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_token_mask=padding_token_mask,
            aux_balance_weight=aux_balance_weight,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits_and_topk=output_router_logits_and_topk,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        all_aux_loss = outputs.all_aux_loss if return_dict else outputs[-1]
        all_aux_loss = torch.mean(torch.cat([l.unsqueeze(0) for l in all_aux_loss], dim=0))
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            aux_loss = self.cur_aux_weight * all_aux_loss
            self.training_steps += 1

            import wandb

            self.input_max_length = max(self.input_max_length, input_ids.shape[1])
            if wandb.run is not None:
                wandb.log({"loss": loss.item(), "aux balance loss": aux_loss.item(), "aux balance weight": self.cur_aux_weight})
                print(
                    f"loss: {loss.item()}, aux balance loss: {aux_loss.item()}, input max length: {self.input_max_length}, self.cur_aux_weight, {self.cur_aux_weight}, aux num: {self.mlp_dynamic_expert_num + self.mlp_fixed_expert_num}"
                )
            else:
                print(
                    f"[no wandb] loss: {loss.item()}, aux balance loss: {aux_loss.item()}, input max length: {self.input_max_length}, self.cur_aux_weight, {self.cur_aux_weight}, aux num: {self.mlp_dynamic_expert_num + self.mlp_fixed_expert_num}"
                )

            loss = loss + aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoEQwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
            all_router_logits=outputs.all_router_logits,
            all_router_top_k=outputs.all_router_top_k,
            all_router_expert_mask=outputs.all_router_expert_mask,
            all_router_weight=outputs.all_router_weight,
            aux_balance_loss=all_aux_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None: 
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs


AutoConfig.register("UniMoEV2_qwen2_vl", UniMoEV2Qwen2VLConfig)
AutoModelForCausalLM.register(UniMoEV2Qwen2VLConfig, UniMoEV2Qwen2VLForConditionalGeneration)

