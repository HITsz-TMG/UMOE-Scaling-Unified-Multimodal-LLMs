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
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchMerger, Qwen2VLVisionBlock, VisionRotaryEmbedding
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None


logger = logging.get_logger(__name__)


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
        # Calculate padding for same output size
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


class PatchEmbed(nn.Module):
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
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, Conv3D)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)])
        self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size)

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

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

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)
