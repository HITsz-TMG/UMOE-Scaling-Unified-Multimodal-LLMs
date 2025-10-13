from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import deepspeed
import torch
import torch.nn.functional as F
from deepspeed import comm as dist
from deepspeed.moe.sharded_moe import _capacity, _one_hot_to_float, einsum, gumbel_rsample
from torch import Tensor

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass

"""
Importing this file will modify:
deepspeed.moe.sharded_moe.MOELayer.forward = gate_forward
deepspeed.moe.sharded_moe.top2gating = top2gating
It removes the AlltoAll communication, allowing Deepspeed MoE to perform inference on a single machine without needing a Deepspeed distributed launch. 
However, ep_size cannot be used (it is recommended to set ep_size=1).
"""



def _AllToAll_forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
    ctx.group = group
    input = input.contiguous()
    return input


def gate_forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

    d_model = input[0].shape[-1]
    reshaped_input = input[0].reshape(-1, d_model)

    if self.use_tutel:
        self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], True)
        S, M = reshaped_input.size(0), reshaped_input.size(1)

        if not hasattr(self, "_tutel_dispatcher"):
            self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
        self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
        dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
    else:
        self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
        dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

    dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)
    expert_output = self.experts(dispatched_input)
    expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, dispatched_input.shape[2], -1)

    if self.use_tutel:
        combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
    else:
        combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)

    a = combined_output.reshape(input[0].size()[:-1] + (-1,))

    return a


def top2gating(
    logits: Tensor, capacity_factor: float, min_capacity: int, drop_tokens: bool = True, ep_group: Union[torch.distributed.ProcessGroup, None] = None, top2_2nd_expert_sampling: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=1)
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    if top2_2nd_expert_sampling:
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits += gumbel_rsample(logits.shape, device=logits.device)

    logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)


    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    exp_counts = torch.sum(mask1 + mask2, dim=0).detach().to(logits.device)

    if drop_tokens:
        capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))
        mask1 *= torch.lt(locations1, capacity)
        mask2 *= torch.lt(locations2, capacity)
    else:
        new_capacity = torch.max(exp_counts)
        capacity = new_capacity


    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)


    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s

    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


deepspeed.moe.sharded_moe.MOELayer.forward = gate_forward
deepspeed.moe.sharded_moe.top2gating = top2gating
deepspeed.moe.sharded_moe._AllToAll.forward = _AllToAll_forward


