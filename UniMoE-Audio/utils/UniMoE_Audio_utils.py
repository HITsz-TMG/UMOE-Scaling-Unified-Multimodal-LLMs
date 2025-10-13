# -*- coding: utf-8 -*-
"""
UniMoE Audio Utilities Module
Author: UniMoE Audio Team
"""

import copy
import glob
import json
import math
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING, Callable

import dac
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from audiotools import AudioSignal
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessor, LogitsProcessorList

import deepspeed
from deepspeed import comm as dist
from deepspeed.moe.sharded_moe import _capacity, _one_hot_to_float, einsum, gumbel_rsample
from torch import Tensor

try:
    import torch_npu
    IS_CUDA = False
except:
    IS_CUDA = True

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass


# =============================================================================
# DAC Utilities
# =============================================================================

class Dac:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        dac_model_dir = os.path.join(base_dir, "dac_model")
        model_path = os.path.join(dac_model_dir, "weights_16khz.pth")
        
        if not os.path.isfile(model_path):
            print(f"DAC model not found at {model_path}, downloading...")
            os.makedirs(dac_model_dir, exist_ok=True)
            downloaded_path = dac.utils.download(model_type="16khz")
            shutil.move(downloaded_path, model_path)
            print(f"DAC model downloaded and saved to {model_path}")
        
        env_path = os.environ.get("DAC_WEIGHTS")
        candidates = []
        if env_path:
            candidates.append(env_path)
        
        candidates.extend([
            model_path, 
            os.path.join(base_dir, "weights_16khz.pth"),
            os.path.join(os.getcwd(), "utils", "dac_model", "weights_16khz.pth"),
            os.path.join(os.getcwd(), "dac_model", "weights_16khz.pth"),
        ])
        
        final_model_path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if not final_model_path:
            searched = "\n - " + "\n - ".join(candidates)
            raise FileNotFoundError(
                "DAC weights not found. Please place weights_16khz.pth in one of the following locations or set DAC_WEIGHTS to an absolute path:" + searched
            )
            
        self.model = dac.DAC.load(final_model_path)
        self.resampler = dict()
        if IS_CUDA:
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("npu")

    def encode(self, audio_path):
        signal = AudioSignal(audio_path)
        if signal.audio_data.shape[1] == 2:
            signal.audio_data = 0.5 * (signal.audio_data[:, :1, :] + signal.audio_data[:, 1:, :])
        signal.to(self.model.device)

        if signal.sample_rate != 16000:
            if not str(signal.sample_rate) in self.resampler:
                self.resampler[str(signal.sample_rate)] = torchaudio.transforms.Resample(signal.sample_rate, 16000)
                if IS_CUDA:
                    self.resampler[str(signal.sample_rate)] = self.resampler[str(signal.sample_rate)].cuda()
                else:
                    self.resampler[str(signal.sample_rate)] = self.resampler[str(signal.sample_rate)].npu()

            signal.audio_data = self.resampler[str(signal.sample_rate)](signal.audio_data)
            signal.sample_rate = 16000

        x = self.model.preprocess(signal.audio_data.to(self.model.device), signal.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)

        codes = codes[0].clone().detach().transpose(0, 1)
        assert codes.shape[1] == 12 and len(codes.shape) == 2
        codes = codes.tolist()

        return codes 

    def decode(self, codes, save_path, min_duration=None):
        assert codes.shape[0] == 1 and codes.shape[1] == 12
        z, _, _ = self.model.quantizer.from_codes(codes.to(self.model.device))
        audio_out = self.model.decode(z)[0].detach().cpu()

        sample_rate = 16000
        duration = audio_out.size(1) / sample_rate
        if min_duration is not None and duration < min_duration:
            padding_duration = min_duration - duration
            padding_samples = int(padding_duration * sample_rate)
            padding = torch.zeros((audio_out.size(0), padding_samples), dtype=audio_out.dtype, device=audio_out.device)
            audio_out = torch.cat((audio_out, padding), dim=1)

        torchaudio.save(save_path, audio_out.detach().cpu(), sample_rate=16000, encoding="PCM_S", bits_per_sample=16)


def build_delay_indices(B: int, T: int, C: int, delay_pattern: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)

    t_idx_BxT = torch.broadcast_to(
        torch.arange(T, dtype=torch.int32)[None, :],
        [B, T],
    )
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)

    b_idx_BxTxC = torch.broadcast_to(
        torch.arange(B, dtype=torch.int32).view(B, 1, 1),
        [B, T, C],
    )
    c_idx_BxTxC = torch.broadcast_to(
        torch.arange(C, dtype=torch.int32).view(1, 1, C),
        [B, T, C],
    )
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)
    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        dim=1,
    ).long()

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(audio_BxTxC: torch.Tensor, pad_value: int, bos_value: int, precomp: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    device = audio_BxTxC.device 
    t_idx_BxTxC, indices_BTCx3 = precomp
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)
    mask_bos = t_idx_BxTxC < 0  
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]  

    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)

    result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))

    return result_BxTxC


def build_revert_indices(B: int, T: int, C: int, delay_pattern: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    device = None
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device)
    t_idx_BT1 = torch.broadcast_to(torch.arange(T, device=device).unsqueeze(0), [B, T])
    t_idx_BT1 = t_idx_BT1.unsqueeze(-1)
    t_idx_BxTxC = torch.minimum(
        t_idx_BT1 + delay_arr.view(1, 1, C),
        torch.tensor(T - 1, device=device),
    )
    b_idx_BxTxC = torch.broadcast_to(torch.arange(B, device=device).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C, device=device).view(1, 1, C), [B, T, C])
    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_idx_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        axis=1,
    ).long()

    return t_idx_BxTxC, indices_BTCx3


def revert_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    precomp: Tuple[torch.Tensor, torch.Tensor],
    T: int,
) -> torch.Tensor:
    t_idx_BxTxC, indices_BTCx3 = precomp
    device = audio_BxTxC.device  
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.size())

    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
    T_tensor = torch.tensor(T, device=device)

    result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)

    return result_BxTxC


def _prepare_audio_prompt(model, audio_prompts: list[torch.Tensor]):
    num_channels = model.config.codec_channels
    audio_bos_value = model.config.codec_bos_value
    delay_pattern = model.config.codec_delay_pattern
    max_delay_pattern = max(delay_pattern)
    batch_size = len(audio_prompts)
    max_len = max(p.shape[0] if p is not None else 0 for p in audio_prompts) + max_delay_pattern + 1
    prefill_steps = []
    prefill = torch.full(
        (batch_size, max_len, num_channels),
        fill_value=-1,
        dtype=torch.int,
        device=model.device,
    )
    prefill[:, 0, :] = audio_bos_value
    for i in range(batch_size):
        prompt = audio_prompts[i]
        if prompt is not None:
            prompt = prompt.to(device=model.device, dtype=torch.int)
            prefill[i, 1 : prompt.shape[0] + 1, :] = prompt
            prefill_steps.append(prompt.shape[0] + 1)
        else:
            prefill_steps.append(1)

    delay_precomp = build_delay_indices(
        B=batch_size,
        T=max_len,
        C=num_channels,
        delay_pattern=delay_pattern,
    )

    delayed_batch = apply_audio_delay(
        audio_BxTxC=prefill,
        pad_value=-1,
        bos_value=audio_bos_value,
        precomp=delay_precomp,
    )

    return delayed_batch, prefill_steps


class DecoderOutput:
    def __init__(self, prefill, prefill_steps, device: torch.device, labels_prefill=None):
        self.generated_tokens = prefill
        self.prefill_steps = prefill_steps
        self.labels_prefill = labels_prefill
        self.device = device

    def get_tokens_at(self, step_from: int, step_to: int = None) -> torch.Tensor:
        if step_to is None:
            step_to = step_from + 1
        return self.generated_tokens[:, step_from:step_to, :].to(self.device)

    def get_labels_at(self, step_from: int, step_to: int = None) -> torch.Tensor:
        if step_to is None:
            step_to = step_from + 1
        if self.labels_prefill is None:
            return None
        return self.labels_prefill[:, step_from:step_to, :].to(self.device)

    def update_one(self, dec_out: torch.Tensor, step: int, apply_mask: bool = False):
        dec_out = dec_out.to(self.generated_tokens.dtype).to(self.generated_tokens.device)
        if apply_mask:
            assert step < self.generated_tokens.shape[1]
            mask = self.generated_tokens[:, step, :] == -1
            self.generated_tokens[:, step, :] = torch.where(mask, dec_out, self.generated_tokens[:, step, :])
        else:
            assert step == self.generated_tokens.shape[1]
            self.generated_tokens = torch.cat((self.generated_tokens, dec_out[:, None, :]), dim=1)


def _generate_output(model, generated_codes: torch.Tensor, lengths_Bx: torch.Tensor) -> list[np.ndarray]:
    num_channels = model.config.codec_channels
    batch_size = generated_codes.shape[0]
    seq_length = generated_codes.shape[1]
    delay_pattern = model.config.codec_delay_pattern
    audio_pad_value = model.config.codec_pad_value
    max_delay_pattern = max(delay_pattern)
    revert_precomp = build_revert_indices(
        B=batch_size,
        T=seq_length,
        C=num_channels,
        delay_pattern=delay_pattern,
    )
    codebook = revert_audio_delay(
        audio_BxTxC=generated_codes,
        pad_value=audio_pad_value,
        precomp=revert_precomp,
        T=seq_length,
    )[:, :-max_delay_pattern, :]

    audios = []
    for i in range(batch_size):
        audios.append(codebook[i, : lengths_Bx[i], :].cpu())

    return audios


# =============================================================================
# DeepSpeed MoE Inference Utilities
# =============================================================================

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


# Apply the modifications to deepspeed
deepspeed.moe.sharded_moe.MOELayer.forward = gate_forward
deepspeed.moe.sharded_moe.top2gating = top2gating
deepspeed.moe.sharded_moe._AllToAll.forward = _AllToAll_forward


# =============================================================================
# Matrix Compression Utilities 
# =============================================================================

def compress_matrix(A: torch.Tensor, mask: torch.Tensor, force_dim: int = None, allow_larger_dim=None) -> torch.Tensor:
    if A.shape[:2] != mask.shape:
        raise ValueError("First two dimensions of A and mask must match.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor.")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError(
            f"mask must only contain 0s and 1s. dtype: {mask.dtype}. "
            f"Invalid elements found at indices: {((mask != 0) & (mask != 1)).nonzero().tolist()} "  # Get indices of elements not 0 AND not 1
            f"with corresponding values: {mask[((mask != 0) & (mask != 1))].tolist()}. "  # Get the values at those indices
            f"\nOriginal mask (showing up to first 20 elements if large):\n{mask.flatten()[:20]}{'...' if mask.numel() > 20 else ''}"
        )

    S, E = mask.shape
    trailing_dims_shape = A.shape[2:]
    num_trailing_dims = len(trailing_dims_shape)
    device = A.device

    ones_per_column = mask.sum(dim=0)
    X = ones_per_column.max().item() if force_dim is None else force_dim

    if X == 0:
        return torch.empty((0, E, *trailing_dims_shape), dtype=A.dtype, device=device)

    sorted_row_indices_2d = torch.argsort(mask.float(), dim=0, descending=True)  
    view_shape_for_indices = (S, E, *((1,) * num_trailing_dims))
    expanded_indices = sorted_row_indices_2d.view(view_shape_for_indices).expand_as(A)

    A_gathered = torch.gather(A, 0, expanded_indices)  

    if X <= A_gathered.shape[0]:
        B_candidate = A_gathered[:X, ...] 
    elif allow_larger_dim or allow_larger_dim is None:
        if allow_larger_dim is None:
            print(f"[Warning compress_matrix] Target dimension X ({X}) is larger than "
                      f"A's original row count S ({S}). Padding B_candidate with zeros.")
        B_candidate = A_gathered 
        zeros_shape = [X - A_gathered.shape[0]] + list(B_candidate.shape[1:])
        B_candidate = torch.cat((B_candidate, torch.zeros(zeros_shape, dtype=B_candidate.dtype, device=B_candidate.device)), dim=0)  # Shape (X_target_dim, E, ...)
    else:
        raise AssertionError(
                f"Target dimension X ({X}) is larger than A's original row count S ({S}) "
                f"and allow_larger_dim is False. Padding is disallowed."
            )
    row_indices_for_B = torch.arange(X, device=device).unsqueeze(1) 
    b_mask_2d = row_indices_for_B < ones_per_column.unsqueeze(0)  
    view_shape_for_b_mask = (X, E, *((1,) * num_trailing_dims))
    B = B_candidate * b_mask_2d.view(view_shape_for_b_mask).to(A.dtype)

    return B


def decompress_matrix(B: torch.Tensor, mask: torch.Tensor, allow_larger_dim=None) -> torch.Tensor:
    if B.shape[1] != mask.shape[1]:
        raise ValueError("B's second dimension and mask's second dimension (E) must match.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor.")
    if not ((mask == 0) | (mask == 1)).all(): 
        raise ValueError("mask must only contain 0s and 1s.")

    S, E = mask.shape
    X = B.shape[0]
    trailing_dims_shape = B.shape[2:]
    num_trailing_dims = len(trailing_dims_shape)
    device = B.device

    if X == 0:  return torch.zeros((S, E, *trailing_dims_shape), dtype=B.dtype, device=device)
    if X <= S: pass
    elif allow_larger_dim or allow_larger_dim is None:
        if allow_larger_dim is None:
                print(f"[Warning decompress_matrix] Input B.shape[0] ({X}) is larger than "
                      f"target A's row count S ({S}). Truncating B to its first {S} rows.")
        B = B[:S, ...]
        X = S
    else:
        raise AssertionError(
                f"Input B.shape[0] ({X}) is larger than target A's row count S ({S}) "
                f"and allow_larger_dim is False. Truncation is disallowed."
            )

    sorted_row_indices_2d = torch.argsort(mask.float(), dim=0, descending=True)  
    target_A_row_indices_2d = sorted_row_indices_2d[:X, :]  
    A_reconstructed = torch.zeros((S, E, *trailing_dims_shape), dtype=B.dtype, device=device)
    view_shape_for_target_indices = (X, E, *((1,) * num_trailing_dims))
    expanded_target_indices = target_A_row_indices_2d.view(view_shape_for_target_indices).expand_as(B)
    A_reconstructed.scatter_(dim=0, index=expanded_target_indices, src=B)

    return A_reconstructed


# =============================================================================
# Qwen2-VL model Utilities 
# =============================================================================

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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.common_types import _size_3_t
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig, Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding, Qwen2_5_VLVisionBlock, Qwen2_5_VLPatchMerger, Qwen2RMSNorm
from transformers.utils import logging

logger = logging.get_logger(__name__)

FAST_INIT = True
if FAST_INIT:
    logger.warning(f"using FAST initial for npu Qwen2_5_vl !!!")


class Conv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._triple(kernel_size)
        self.stride = self._triple(stride)
        self.dilation = self._triple(dilation)
        self.padding_mode = padding_mode
        self.groups = groups

        if isinstance(padding, str):
            if padding == "valid":
                self.padding = (0, 0, 0)
            elif padding == "same":
                self.padding = self._calculate_same_padding()
            else:
                raise ValueError(f"Unsupported padding mode: {padding}")
        else:
            self.padding = self._triple(padding)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups * self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain("conv3d"))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def _triple(self, x: _size_3_t) -> Tuple[int, int, int]:
        if isinstance(x, int):
            return (x, x, x)
        if len(x) == 3:
            return x
        raise ValueError(f"Invalid 3D parameter: {x}")

    def _calculate_same_padding(self) -> Tuple[int, int, int]:
        def get_pad(size, kernel, stride, dilation):
            return ((size - 1) * stride + dilation * (kernel - 1)) // 2

        pad_d = get_pad(1, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_h = get_pad(1, self.kernel_size[1], self.stride[1], self.dilation[1])
        pad_w = get_pad(1, self.kernel_size[2], self.stride[2], self.dilation[2])
        return (pad_d, pad_h, pad_w)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = input.shape
        (pad_d, pad_h, pad_w) = self.padding

        if self.padding_mode != "zeros":
            input = F.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode=self.padding_mode)
            pad_d = pad_h = pad_w = 0

        D = depth + 2 * pad_d
        H = height + 2 * pad_h
        W = width + 2 * pad_w

        D_out = (D - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1

        depth_indices = []
        for t in range(D_out):
            start = t * self.stride[0]
            for i in range(self.kernel_size[0]):
                idx = start + i * self.dilation[0]
                depth_indices.append(idx)

        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))
        slices = input_padded[:, :, depth_indices, :, :]
        slices = slices.view(batch_size, channels, D_out, self.kernel_size[0], H, W).permute(0, 2, 1, 3, 4, 5)
        slices = slices.contiguous().view(batch_size * D_out, channels * self.kernel_size[0], H, W)
        output = F.conv2d(slices, self.weight, bias=None, stride=(self.stride[1], self.stride[2]), padding=(0, 0), dilation=(self.dilation[1], self.dilation[2]), groups=self.groups)

        C_out = self.out_channels
        H_out = output.shape[2]
        W_out = output.shape[3]
        output = output.view(batch_size, D_out, C_out, H_out, W_out)
        output = output.permute(0, 2, 1, 3, 4)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)

        return output

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0, 0, 0):
            s += ", padding={padding}"
        if self.dilation != (1, 1, 1):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = Conv3D(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2_5_VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.get_text_config().initializer_range
        if not FAST_INIT:
            if isinstance(module, (nn.Linear, Conv3D)):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, Qwen2RMSNorm):
                module.weight.data.fill_(1.0)

            
class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    config_class = Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states
