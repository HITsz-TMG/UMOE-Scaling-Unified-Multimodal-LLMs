from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from deepspeed.utils import groups, log_dist


# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from deepspeed.moe.mappings import drop_tokens, gather_tokens


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs_raw: torch.Tensor):
        ishape = inputs_raw.shape
        inputs = inputs_raw.view(-1,ishape[-1])
        gate_logits = self.gate(inputs)
        # l_aud
        gates = F.softmax(gate_logits, dim=1)
        indices1_s = torch.argmax(gates, dim=1)
        num_experts = int(gates.shape[1])
        mask1 = F.one_hot(indices1_s, num_classes=num_experts)

        # Compute l_aux
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.float(), dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts
        # print("laux",l_aux)

        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        results_out = results.view(ishape)
        return results_out,l_aux,gate_logits,"moe"