# Adopted from https://github.com/mistralai. Below is the original copyright:

import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from torch import nn



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
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        results_out = results.view(ishape)
        return results_out
