import copy
import os
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import deepspeed
from deepspeed import comm as dist
from deepspeed.utils import groups, log_dist
from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.moe.sharded_moe import FIRST_ALLTOALL_TIMER, MOE_TIMER, SECOND_ALLTOALL_TIMER, _AllToAll, einsum, gumbel_rsample
from transformers.activations import ACT2FN

from .UniMoE_Audio_utils import compress_matrix, decompress_matrix

class AudioSharedExpertMLP(nn.Module):
    """
    Shared expert MLP for UniMoE-Audio model.
    Handles common audio feature transformations across all tokens.
    """
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


class AudioDynamicExpertMLP(nn.Module):
    """
    Dynamic expert MLP for UniMoE-Audio model.
    Specialized for adaptive audio feature processing based on content.
    """
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


class AudioNullExpertMLP(nn.Module):
    """
    Null expert MLP for UniMoE-Audio model.
    Returns zero output for tokens that don't require expert processing.
    """
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_state):
        return torch.zeros_like(hidden_state, dtype=hidden_state.dtype, device=hidden_state.device)


class AudioMoERoutingFunction(torch.autograd.Function):
    """
    Custom autograd function for UniMoE-Audio routing computation.
    Implements differentiable expert selection with gradient flow control.
    """
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
    def backward(ctx, grad_at_output: torch.Tensor):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors
        grad_at_output = grad_at_output * multiplier
        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
            )
        return (grad_at_scores_expaned, None, None, None, None,)


def audio_sparse_expert_mixer(scores, top_k, jitter_eps, training):
    """
    Sparse expert mixing function for UniMoE-Audio.
    Implements adaptive expert selection with noise injection for training.
    """
    masked_scores = scores
    multiplier_list = []
    selected_experts_list = []

    for _ in range(top_k):
        with torch.no_grad():
            mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
            factor = scores.abs().clamp(min=mask_logits_threshold.abs())
            mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

        masked_gates = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))

        if training:
            noise = gumbel_rsample(masked_gates.shape, device=masked_gates.device)
            new_masked_gates = masked_gates + noise
            selected_experts = (new_masked_gates).max(dim=-1)[1].unsqueeze(-1)
        else:
            selected_experts = max_ind

        masked_gates = torch.softmax(masked_gates, dim=-1)
        multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

        if training:
            max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
            mask_for_one = torch.logical_or(
                selected_experts == max_ind,
                torch.rand_like(max_scores) > 0.75,
            )
            mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

            multiplier = AudioMoERoutingFunction.apply(
                scores,
                multiplier_o,
                selected_experts,
                masked_gates,
                mask_for_one,
            )
        else:
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


def audio_dynamic_expert_selection(logits, top_p):
    """
    Dynamic expert selection for UniMoE-Audio based on cumulative probability threshold.
    Adapts the number of experts based on audio content complexity.
    """
    dynamic_scores = torch.softmax(logits, dim=-1)
    dynamic_scores_sorted, _ = torch.sort(dynamic_scores, dim=-1, descending=True)
    dynamic_scores_cumsum = dynamic_scores_sorted.cumsum(dim=-1)
    dynamic_top_k = (~(dynamic_scores_cumsum >= top_p)).sum(dim=-1)
    dynamic_top_k = dynamic_top_k + 1
    return dynamic_top_k


def _audio_expert_capacity(num_tokens, num_experts, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    """Calculate expert capacity for UniMoE-Audio based on token distribution and capacity factor."""
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


def calculate_audio_global_routing_weight(
    expert_mask: torch.Tensor,
    full_router_logits: torch.Tensor,
    mlp_dynamic_expert_num: int,
    routing_weights: torch.Tensor,
):
    """
    Calculate global routing weights for UniMoE-Audio combining dynamic and fixed expert weights.
    Optimized for audio generation tasks.
    """
    global_weight = torch.softmax(full_router_logits.masked_fill(expert_mask == 0, float("-inf")), dim=-1)
    global_dynamic_weight = global_weight[:, :mlp_dynamic_expert_num]
    global_fixed_weight = global_weight[:, mlp_dynamic_expert_num:]
    global_dynamic_weight = routing_weights * global_dynamic_weight.sum(-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1])
    global_weight = torch.cat((global_dynamic_weight, global_fixed_weight), dim=-1)
    return global_weight


class UniMoEAudioSparseMoeBlock(nn.Module):
    """
    UniMoE-Audio Sparse Mixture of Experts block with dynamic routing and expert selection.
    Optimized for audio generation tasks with efficient sparse operations and capacity management.
    """

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

        self.ignore_differentiable_router = config.ignore_differentiable_router
        if self.ignore_differentiable_router:
            print("ignore_differentiable_router is True, will not use router_logits !!!")

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.fixed_real_moe = nn.ModuleList([AudioSharedExpertMLP(config) for _ in range(self.mlp_fixed_expert_num)])
        self.dynamic_real_moe = UniMoEAudioMoE(config, AudioDynamicExpertMLP(config), self.mlp_dynamic_real_expert_num, config.ep_size)

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
            dynamic_top_k = audio_dynamic_expert_selection(dynamic_router_logits, self.mlp_dynamic_top_p)
        else:
            dynamic_top_k = torch.full((dynamic_router_logits.shape[0],), self.mlp_dynamic_top_k, dtype=torch.int, device=dynamic_router_logits.device)

        expert_mask = torch.zeros((batch_size * sequence_length, self.num_experts), dtype=torch.int, device=hidden_states.device)

        routing_weights = torch.zeros((batch_size * sequence_length, self.mlp_dynamic_expert_num), dtype=hidden_states.dtype, device=hidden_states.device)
        for top_k in range(1, self.mlp_dynamic_expert_num + 1):
            group_idx = torch.nonzero(dynamic_top_k == top_k, as_tuple=True)[0]
            if len(group_idx) == 0:
                continue

            dynamic_group_logits = dynamic_router_logits[group_idx]
            group_routing_weights, group_selected_experts = audio_sparse_expert_mixer(
                dynamic_group_logits,
                top_k=top_k,
                jitter_eps=self.router_jitter_noise,
                training=self.training and not self.ignore_differentiable_router,
            )

            group_expert_mask = torch.nn.functional.one_hot(group_selected_experts, num_classes=self.num_experts)
            group_expert_mask = group_expert_mask.sum(dim=1)

            group_weight = torch.zeros((len(group_idx), self.mlp_dynamic_expert_num), dtype=hidden_states.dtype, device=hidden_states.device)
            group_weight.scatter_(dim=-1, index=group_selected_experts, src=group_routing_weights)
            routing_weights.index_add_(0, group_idx, group_weight)

            expert_mask.index_add_(0, group_idx, group_expert_mask.to(expert_mask.dtype))

        routing_weights = routing_weights / (routing_weights.sum(dim=-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1]) + 1e-6)

        if attention_mask is not None:
            attention_mask = attention_mask.to(expert_mask.dtype).view(-1).unsqueeze(-1).expand(-1, self.num_experts)
            expert_mask = expert_mask * attention_mask

        if self.mlp_dynamic_expert_num < self.num_experts:
            expert_mask[:, self.mlp_dynamic_expert_num :] = 1

        aux_loss = audio_load_balancing_loss_func(
            expert_mask=expert_mask,
            mlp_dynamic_expert_num=self.mlp_dynamic_expert_num,
            global_weight=None,
            full_router_logits=full_router_logits,
            routing_weights=routing_weights,
            aux_balance_weight=aux_balance_weight,
        )

        if self.token_drop:
            expert_mask_dtype = expert_mask.dtype
            capacity = _audio_expert_capacity(batch_size * sequence_length, self.mlp_dynamic_expert_num, torch.tensor(self.capacity_factor), torch.tensor(self.min_capacity))
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
            global_weight = calculate_audio_global_routing_weight(expert_mask, full_router_logits, self.mlp_dynamic_expert_num, routing_weights)
        else:
            global_weight = routing_weights

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


def audio_load_balancing_loss_func(
    expert_mask: torch.Tensor,
    mlp_dynamic_expert_num: int,
    global_weight: Optional[torch.Tensor] = None,
    full_router_logits: Optional[torch.Tensor] = None,
    routing_weights: Optional[torch.Tensor] = None,
    aux_balance_weight: Optional[torch.Tensor] = None,
) -> float:
    """Calculate load balancing loss for UniMoE-Audio expert routing to encourage balanced usage."""
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


class AudioExperts(deepspeed.moe.experts.Experts):
    """Custom Audio experts class extending DeepSpeed MoE experts with additional functionality."""
    
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


class AudioMOELayer(deepspeed.moe.sharded_moe.MOELayer):
    """Custom Audio MoE layer extending DeepSpeed MOELayer with matrix compression optimization."""
    
    def __init__(
        self,
        experts: nn.Module,
        ep_group_name,
        ep_size,
        num_local_experts: int,
        use_tutel: bool = False,
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
        compres_hidden_states = compress_matrix(compres_hidden_states, expert_mask, force_dim=capacity, allow_larger_dim=True)  # [C, expert_num, d_model]
        compres_expert_mask = compress_matrix(expert_mask, expert_mask, force_dim=capacity, allow_larger_dim=True)
        dispatched_input = einsum("ce,cem->ecm", compres_expert_mask, compres_hidden_states)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).start()

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


class UniMoEAudioMoE(deepspeed.moe.layer.MoE):
    """Custom Audio MoE class extending DeepSpeed MoE with configuration and parallelism setup."""
    
    def __init__(self, config, expert, num_experts, ep_size, moe_name_prefix="ep_size"):
        super(deepspeed.moe.layer.MoE, self).__init__()
        self.enable_expert_tensor_parallelism = config.enable_expert_tensor_parallelism
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.expert_group_name = f"{moe_name_prefix}_{self.ep_size}"
        self.num_local_experts = self.num_experts // self.ep_size
        log_dist(f"Creating MoE layer with num_experts: {self.num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}", [0])
        experts = AudioExperts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = AudioMOELayer(experts, self.expert_group_name, self.ep_size, self.num_local_experts)

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
