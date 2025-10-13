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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from transformers.configuration_utils import PretrainedConfig, layer_type_validation

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import (
    ModelOutput,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    Qwen2RMSNorm,
    Qwen2_5_VLRotaryEmbedding,
)
from .UniMoE_Audio_core import UniMoEAudioSparseMoeBlock
from .UniMoE_Audio_utils import Qwen2_5_VisionTransformerPretrainedModel

logger = logging.get_logger(__name__)

FAST_INIT = True
if FAST_INIT:
    logger.warning(f"using FAST initial for Grin Qwen2_vl !!!")

class Qwen2_5_VLMoETextConfig(Qwen2_5_VLTextConfig):
    model_type = "qwen2_5_vl_moe_text"

    def __init__(
        self,
        mlp_dynamic_expert_num=4,
        mlp_dynamic_null_expert_num=0,
        mlp_dynamic_top_p=0.7,
        mlp_dynamic_top_k=2,
        mlp_fixed_expert_num=2,
        dynamic_intermediate_size=8960,
        shared_intermediate_size=8960,
        ignore_differentiable_router=False,
        enable_expert_tensor_parallelism: bool = False,
        ep_size=1,
        fixed_ep_size=1,
        router_jitter_noise=0.01,
        input_jitter_noise=0.01,
        token_drop=False,
        drop_policy: str = "probs", 
        min_capacity: int = 8,
        capacity_factor: float = 1.0,
        fp32_gate=True,
        avg_hidden_states_last=False,
        drop_token_num_print=True,
        l_aux_weight=0,
        min_l_aux_weight=0,
        l_aux_weight_decay_steps=1,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.mlp_dynamic_expert_num = mlp_dynamic_expert_num
        self.mlp_dynamic_top_p = mlp_dynamic_top_p
        self.mlp_dynamic_top_k = mlp_dynamic_top_k
        self.mlp_fixed_expert_num = mlp_fixed_expert_num
        self.mlp_dynamic_null_expert_num = mlp_dynamic_null_expert_num
        self.dynamic_intermediate_size = dynamic_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.ignore_differentiable_router = ignore_differentiable_router
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
        self.min_l_aux_weight = min_l_aux_weight
        self.l_aux_weight_decay_steps = l_aux_weight_decay_steps


class UniAudioRVQQwen2_5VLMoEConfig(PretrainedConfig):
    model_type = "uni_audio_rvq_qwen2_5vl_moe"
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": Qwen2_5_VLMoETextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        codec_vocab_size=1028,
        codec_delay_pattern=[0, 8, 9, 10, 11, 12, 13, 14, 15],
        codec_channels=9,
        codec_eos_value=1024,
        codec_pad_value=1025,
        codec_bos_value=1026,
        codec_placeholder_value=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.codec_vocab_size = codec_vocab_size
        self.codec_delay_pattern = codec_delay_pattern
        self.codec_channels = codec_channels
        self.codec_eos_value = codec_eos_value
        self.codec_pad_value = codec_pad_value
        self.codec_bos_value = codec_bos_value
        self.codec_placeholder_value = codec_placeholder_value

        super().__init__(**kwargs)

@dataclass
class MoEQwen2_5VLCausalLMOutputWithPast(ModelOutput):
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


class Qwen2_5_VLMoEDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2_5_VLMoETextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)
        self.mlp = UniMoEAudioSparseMoeBlock(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aux_balance_weight: Optional[torch.Tensor] = None, 
        padding_token_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits_and_topk: Optional[bool] = False, 
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, 
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
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
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits, router_top_k, router_expert_mask, router_weight, aux_loss = self.mlp(hidden_states, padding_token_mask, aux_balance_weight)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits_and_topk:
            outputs += (router_logits,)
            outputs += (router_top_k,)
        outputs += (router_expert_mask,)
        outputs += (router_weight,)
        outputs += (aux_loss,)

        return outputs


class Qwen2_5_VLMoEPreTrainedModel(PreTrainedModel):
    config_class = UniAudioRVQQwen2_5VLMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLMoEDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if FAST_INIT:
            if isinstance(module, UniMoEAudioSparseMoeBlock):
                module.gate.weight.data.normal_(mean=0.0, std=std)
                if module.gate.bias is not None:
                    module.gate.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        else:
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, Qwen2RMSNorm):
                module.weight.data.fill_(1.0)


class Qwen2_5_VLMoETextModel(Qwen2_5_VLMoEPreTrainedModel):
    config_class = Qwen2_5_VLMoETextConfig
    def __init__(self, config: Qwen2_5_VLMoETextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLMoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        aux_balance_weight: Optional[torch.Tensor] = None, 
        padding_token_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits_and_topk: Optional[bool] = None, 
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

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

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                aux_balance_weight=aux_balance_weight, 
                padding_token_mask=padding_token_mask, 
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits_and_topk=output_router_logits_and_topk, 
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

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

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states, 
                    past_key_values, 
                    all_hidden_states, 
                    all_self_attns, 
                    all_router_logits, 
                    all_router_top_k, 
                    all_router_expert_mask, 
                    all_router_weight, 
                    all_aux_loss] 
                    if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            all_router_logits=all_router_logits,
            all_router_top_k=all_router_top_k,
            all_router_expert_mask=all_router_expert_mask,
            all_router_weight=all_router_weight,
            all_aux_loss=all_aux_loss,
        )


class UniAudioRVQQwen2_5VLMoEForConditionalGeneration(Qwen2_5_VLMoEPreTrainedModel):
    base_model_prefix = ""
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    config_class = UniAudioRVQQwen2_5VLMoEConfig
    _checkpoint_conversion_mapping = {
        "^visual": "visual",
        r"^model(?!\.(language_model|visual))": "language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config, attn_implementation=config._attn_implementation)
        self.language_model = Qwen2_5_VLMoETextModel._from_config(config.text_config)
        self.rope_deltas = None 
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.l_aux_weight = config.text_config.l_aux_weight
        self.min_l_aux_weight = config.text_config.min_l_aux_weight
        self.l_aux_weight_decay_steps = max(1, config.text_config.l_aux_weight_decay_steps)
        self.input_max_length = 0
        self.training_steps = 0
        self.num_channels = config.codec_channels
        self.codec_vocab_size = config.codec_vocab_size
        self.codec_embed_tokens = nn.ModuleList(
            [nn.Embedding(self.codec_vocab_size, config.text_config.hidden_size) for embed_idx in range(self.num_channels)])
        self.codec_placeholder_value = config.codec_placeholder_value
        self.codec_head = nn.Linear(config.text_config.hidden_size, self.num_channels * self.codec_vocab_size, bias=False)
        self.post_init()

    @property
    def cur_aux_weight(self):
        if self.training_steps >= self.l_aux_weight_decay_steps:
            return self.min_l_aux_weight
        return self.l_aux_weight - (self.l_aux_weight - self.min_l_aux_weight) / self.l_aux_weight_decay_steps * self.training_steps

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
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
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
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

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

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
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds


    def codec_embedding(self, codec_input_ids):
        x = None
        for i in range(self.num_channels):
            channel_tokens = codec_input_ids[..., i]
            channel_embed = self.codec_embed_tokens[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed
        return x

    def calculate_input_embedding(self, input_ids, codec_input_ids):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if codec_input_ids is not None:
            codec_input_embeds = self.codec_embedding(codec_input_ids)
            
            codec_mask = (input_ids == self.codec_placeholder_value).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(codec_mask, codec_input_embeds)
        return inputs_embeds

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        codec_input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        codec_labels: Optional[torch.LongTensor] = None,
        aux_balance_weight: Optional[torch.LongTensor] = None,
        padding_token_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits_and_topk: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,

    ) -> Union[Tuple, MoEQwen2_5VLCausalLMOutputWithPast]:
        return_dict = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.calculate_input_embedding(input_ids, codec_input_ids)
            
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)

            if input_ids is None:
                image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                image_mask = image_mask.all(-1)
            else:
                image_mask = input_ids == self.config.image_token_id

            n_image_tokens = (image_mask).sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            n_image_features = image_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0)

            if input_ids is None:
                video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                video_mask = video_mask.all(-1)
            else:
                video_mask = input_ids == self.config.video_token_id

            n_video_tokens = (video_mask).sum()
            n_video_features = video_embeds.shape[0]
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas

            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if aux_balance_weight is not None:
            aux_balance_weight = attention_mask * aux_balance_weight

        if padding_token_mask is None:
            padding_token_mask = attention_mask.bool()

        outputs = self.language_model(
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
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()
        codec_logits = self.codec_head(hidden_states).float()  
        codec_logits = codec_logits.view((logits.shape[0], logits.shape[1], self.num_channels, self.codec_vocab_size))

        loss = None
        if labels is not None:

            all_aux_loss = outputs.all_aux_loss if return_dict else outputs[-1]
            all_aux_loss = torch.mean(torch.cat([l.unsqueeze(0) for l in all_aux_loss], dim=0))
            aux_loss = self.cur_aux_weight * all_aux_loss
            self.training_steps += 1
            codec_loss = None

            if codec_labels is not None:
                for i in range(self.num_channels):
                    channel_logits = codec_logits[:, :, i].float()
                    channel_labels = codec_labels[:, :, i]
                    shift_channel_logits = channel_logits[..., :-1, :].contiguous()
                    shift_channel_labels = channel_labels[..., 1:].contiguous()

                    if i!= 0 and (shift_channel_labels != -100).sum() == 0:
                        continue

                    loss_fct = CrossEntropyLoss()
                    shift_channel_logits = shift_channel_logits.view(-1, self.codec_vocab_size)
                    shift_channel_labels = shift_channel_labels.view(-1)
                    shift_channel_labels = shift_channel_labels.to(shift_channel_logits.device)
                    channel_loss = loss_fct(shift_channel_logits, shift_channel_labels)
                    codec_loss = channel_loss if codec_loss is None else codec_loss + channel_loss

            loss = codec_loss + aux_loss

            import wandb
            self.input_max_length = max(self.input_max_length, input_ids.shape[1])

            if wandb.run is not None:
                wandb.log({"final loss": loss.item(), "codec loss": codec_loss.item(), "weighted aux balance loss": aux_loss.item(), "aux balance weight": self.cur_aux_weight})
                print(f"final loss: {loss.item():.4g}, codec loss: {codec_loss.item():.4g}, weighted aux balance loss: {aux_loss.item():.4g}, aux balance weight: {self.cur_aux_weight:.2g}, input max length: {self.input_max_length}")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoEQwen2_5VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_router_logits=outputs.all_router_logits,
            all_router_top_k=outputs.all_router_top_k,
            all_router_expert_mask=outputs.all_router_expert_mask,
            all_router_weight=outputs.all_router_weight,
            aux_balance_loss=all_aux_loss,
        )

    @staticmethod
    def _sample_next_token(
        logits_BCxV: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        audio_eos_value: int,
    ) -> torch.Tensor:
        if temperature == 0.0:
            return torch.argmax(logits_BCxV, dim=-1)

        logits_BCxV = logits_BCxV / temperature

        if audio_eos_value is not None and audio_eos_value >= 0:
            top_logit_indices_BC = torch.argmax(logits_BCxV, dim=-1)
            eos_not_highest_mask_BC = top_logit_indices_BC != audio_eos_value
            mask_eos_unless_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
            mask_eos_unless_highest_BCxV[eos_not_highest_mask_BC, audio_eos_value] = True
            logits_BCxV = logits_BCxV.masked_fill(mask_eos_unless_highest_BCxV, -torch.inf)

        if top_k is not None:
            _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=top_k, dim=-1)
            mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
            mask = mask.scatter(dim=-1, index=top_k_indices_BCxV, value=False)
            logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

        if top_p < 1.0:
            probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
            sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
            cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

            sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
            sorted_indices_to_remove_BCxV = torch.roll(sorted_indices_to_remove_BCxV, shifts=1, dims=-1)
            sorted_indices_to_remove_BCxV[..., 0] = torch.zeros_like(sorted_indices_to_remove_BCxV[..., 0])

            indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
            indices_to_remove_BCxV = indices_to_remove_BCxV.scatter(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
            logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

        final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

        sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
        sampled_indices_C = sampled_indices_BC.squeeze(-1)
        return sampled_indices_C

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,
        model_kwargs,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample=True,
        eos_prob_mul_factor=1.0,
        labels_Bx1xC=None,
        use_cache=True,
        enable_eos=True,
    ) -> torch.Tensor:
        B = tokens_Bx1xC.shape[0]
        audio_eos_value = self.config.codec_eos_value
        attention_mask = model_kwargs["attention_mask"]  
        cache_position = model_kwargs["cache_position"]  
        past_key_values = model_kwargs["past_key_values"]
        input_ids = model_kwargs["input_ids"]
        codec_input_ids = model_kwargs["codec_input_ids"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -tokens_Bx1xC.shape[1] :]
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        tokens_Bx1xC = tokens_Bx1xC.repeat_interleave(2, dim=0)
        codec_input_ids = torch.cat((codec_input_ids, tokens_Bx1xC), dim=1) if codec_input_ids is not None else tokens_Bx1xC.clone()
        input_ids = torch.cat((input_ids, torch.ones(input_ids.shape[0], 1).to(input_ids) * self.codec_placeholder_value), dim=-1)

        if use_cache:
            codec_input_embeds = self.codec_embedding(tokens_Bx1xC)
            outputs = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=codec_input_embeds,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=cache_position,
            )

        else:
            batch_codec_input_ids = codec_input_ids.contiguous().view(-1, self.num_channels)

            inputs_embeds = self.calculate_input_embedding(input_ids, batch_codec_input_ids)
            outputs = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=attention_mask.long().cumsum(-1) - 1,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=None,
            )

        last_hidden_state = outputs.last_hidden_state
        codec_logits = self.codec_head(last_hidden_state).float()
        codec_logits = codec_logits.view((codec_logits.shape[0], codec_logits.shape[1], self.num_channels, self.codec_vocab_size))
        model_kwargs["past_key_values"] = outputs.past_key_values
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1 
        model_kwargs["input_ids"] = input_ids
        model_kwargs["codec_input_ids"] = codec_input_ids

        logits_Bx1xCxV = codec_logits[: , -1:].clone()
        logits_last_2BxCxV = logits_Bx1xCxV[:, -1]
        logits_last_Bx2xCxV = logits_last_2BxCxV.view(B, 2, *logits_last_2BxCxV.shape[1:])
        if cfg_scale != 0:
            uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :] 
            cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]  
            logits_BxCxV = cond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)
        else:
            logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]  

        if enable_eos:
            logits_BxCxV[:, :, audio_eos_value + 1 :] = torch.full_like(
                logits_BxCxV[:, :, audio_eos_value + 1 :],
                fill_value=-torch.inf,
            )  
            logits_BxCxV[:, 1:, audio_eos_value:] = torch.full_like(
                logits_BxCxV[:, 1:, audio_eos_value:],
                fill_value=-torch.inf,
            ) 
        else:
            logits_BxCxV[:, :, audio_eos_value:] = torch.full_like(
                logits_BxCxV[:, :, audio_eos_value:],
                fill_value=-torch.inf,
            )  


        logits_BxCxV[:, 0, audio_eos_value] *= torch.tensor(eos_prob_mul_factor, device=self.device)

        if labels_Bx1xC is not None:
            codec_labels = model_kwargs["codec_labels"]
            labels = model_kwargs["labels"]
            new_labels_Bx1xC = labels_Bx1xC.clone()
            new_labels_Bx1xC[new_labels_Bx1xC > audio_eos_value] = -100
            tmp_labels_Bx1xC = new_labels_Bx1xC[:, :, 1:].clone()
            tmp_labels_Bx1xC[tmp_labels_Bx1xC >= audio_eos_value] = -100
            new_labels_Bx1xC[:, :, 1:] = tmp_labels_Bx1xC

            codec_labels = torch.cat((codec_labels, new_labels_Bx1xC), dim=1)
            labels = torch.cat((labels, torch.ones((labels.shape[0], 1), device=labels.device, dtype=labels.dtype) * -100), dim=1)

            codec_loss = None
            for i in range(self.num_channels):
                shift_channel_logits = logits_BxCxV[:, i].float()
                shift_channel_labels = new_labels_Bx1xC[:, 0, i]

                if i!= 0 and (shift_channel_labels != -100).sum() == 0:
                    continue

                loss_fct = CrossEntropyLoss()
                shift_channel_labels = shift_channel_labels.to(shift_channel_logits.device)
                channel_loss = loss_fct(shift_channel_logits, shift_channel_labels)

                loss_weight = 3 if i == 0 else 1
                channel_loss = channel_loss * loss_weight

                codec_loss = channel_loss if codec_loss is None else codec_loss + channel_loss

            print(f"golden loss: {codec_loss}")

            model_kwargs["codec_labels"] = codec_labels
            model_kwargs["labels"] = labels


        flat_logits_BCxV = logits_BxCxV.reshape(B * self.num_channels, -1)
        if do_sample:
            pred_BC = self._sample_next_token(
                flat_logits_BCxV.float(),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                audio_eos_value=audio_eos_value,
            )
        else:
            pred_BC = torch.argmax(flat_logits_BCxV, dim=1)

        pred_BxC = pred_BC.view(B, self.num_channels)

        return pred_BxC, model_kwargs

    def generate(
        self,
        input_ids,
        attention_mask,
        dec_output,
        max_tokens,
        min_tokens=None,
        codec_input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.2,
        top_p: float = 0.95,
        cfg_filter_top_k: int = 45,
        eos_prob_mul_factor: float = 0.8,
        do_sample: bool = True,
        debug_guidance_step: int = 0,
        use_cache=True,
    ):
        if codec_input_ids is not None:
            assert use_cache 
        batch_size = input_ids.shape[0] // 2
        audio_eos_value = self.config.codec_eos_value
        audio_pad_value = self.config.codec_pad_value
        delay_pattern = self.config.codec_delay_pattern
        max_delay_pattern = max(delay_pattern)
        delay_pattern_Cx = torch.tensor(delay_pattern, device=self.device, dtype=torch.long)

        dec_step = min(dec_output.prefill_steps) - 1

        eos_detected_Bx = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        eos_countdown_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        finished_step_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

        bos_over = False
        model_kwargs = dict(attention_mask=attention_mask, use_cache=True)
        model_kwargs["past_key_values"] = DynamicCache()
        model_kwargs["cache_position"] = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        attention_mask = model_kwargs["attention_mask"]  
        past_key_values = model_kwargs["past_key_values"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        cache_position = torch.arange(0, input_ids.shape[-1], device=input_ids.device)
        inputs_embeds = self.calculate_input_embedding(input_ids, codec_input_ids)
        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=True,
            output_attentions=False,  
            output_hidden_states=False, 
            return_dict=True,
            cache_position=cache_position,
        )

        model_kwargs["input_ids"] = input_ids
        model_kwargs["codec_input_ids"] = None
        model_kwargs["labels"] = torch.ones_like(input_ids[1::2]) * -100
        labels_Bx1xC = dec_output.get_labels_at(0)
        if labels_Bx1xC is not None:
            model_kwargs["codec_labels"] = (torch.ones_like(input_ids[1::2]) * -100).unsqueeze(-1).expand(-1, -1, self.num_channels)
            assert (labels_Bx1xC != self.config.codec_bos_value).sum() == 0
            labels_Bx1xC = torch.full_like(labels_Bx1xC, -100)
            model_kwargs["codec_labels"] = torch.cat((model_kwargs["codec_labels"], labels_Bx1xC), dim=1)
        model_kwargs["past_key_values"] = outputs.past_key_values
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        while dec_step < max_tokens:
            if (eos_countdown_Bx == 0).all():
                break

            current_step_idx = dec_step + 1
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step)
            labels_Bx1xC = dec_output.get_labels_at(dec_step + 1)

            pred_BxC, model_kwargs = self._decoder_step(
                tokens_Bx1xC=tokens_Bx1xC,
                model_kwargs=model_kwargs,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                top_k=cfg_filter_top_k,
                do_sample=do_sample,
                eos_prob_mul_factor=eos_prob_mul_factor,
                labels_Bx1xC=labels_Bx1xC,
                use_cache=use_cache,
                enable_eos=(min_tokens is None or dec_step >= min_tokens),
            )
            if labels_Bx1xC is not None and (dec_step < debug_guidance_step or debug_guidance_step==-1):
                pred_BxC = labels_Bx1xC[:, 0]

            active_mask_Bx = eos_countdown_Bx != 0
            eos_trigger_Bx = torch.zeros_like(active_mask_Bx)
            if active_mask_Bx.any():
                is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (pred_BxC[active_mask_Bx, 0] == audio_eos_value)
                is_max_len = current_step_idx >= max_tokens - max_delay_pattern
                eos_trigger_Bx[active_mask_Bx] = is_eos_token | is_max_len
            eos_detected_Bx |= eos_trigger_Bx
            start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)
            if start_countdown_mask_Bx.any():
                eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
                finished_step_Bx[start_countdown_mask_Bx] = current_step_idx

            padding_mask_Bx = eos_countdown_Bx > 0
            if padding_mask_Bx.any():
                pred_active_BxC = pred_BxC[padding_mask_Bx].clone()
                countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
                step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
                step_after_eos_Bx_ = step_after_eos_Bx.unsqueeze(1)
                delay_pattern_Cx_ = delay_pattern_Cx.unsqueeze(0)
                eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
                pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
                pred_active_BxC[eos_mask_NxC] = audio_eos_value
                pred_active_BxC[pad_mask_NxC] = audio_pad_value
                pred_BxC[padding_mask_Bx] = pred_active_BxC
                eos_countdown_Bx[padding_mask_Bx] -= 1

            if not bos_over:
                bos_over = all(current_step_idx - prefill_step >= max_delay_pattern for prefill_step in dec_output.prefill_steps)

            dec_output.update_one(pred_BxC, current_step_idx, not bos_over)
            dec_step += 1

        final_step = dec_step + 1
        finished_step_Bx[finished_step_Bx == -1] = final_step - max_delay_pattern
        prefill_steps_tensor = torch.tensor(dec_output.prefill_steps, device=self.device)
        lengths_Bx = finished_step_Bx - prefill_steps_tensor
        lengths_Bx = torch.clamp(lengths_Bx, min=0)
        max_len = lengths_Bx.max().item() + max_delay_pattern

        if max_len > 0:
            num_channels = self.num_channels
            generated_codes = torch.full(
                (batch_size, max_len, num_channels),
                fill_value=audio_pad_value,
                dtype=torch.long,
                device=self.device,
            )

            for i in range(batch_size):
                start_step = dec_output.prefill_steps[i]
                actual_len = lengths_Bx[i].item() + max_delay_pattern
                if actual_len > 0:
                    tokens_to_copy = dec_output.generated_tokens[i, start_step : start_step + actual_len, :]
                    generated_codes[i, :actual_len, :] = tokens_to_copy

            return generated_codes, lengths_Bx
        else:
            print("Warning: Nothing generated for any sequence in the batch.")
            return None, None

AutoConfig.register("qwen2_5_vl_moe_text", Qwen2_5_VLMoETextConfig)
AutoModelForCausalLM.register(Qwen2_5_VLMoETextConfig, Qwen2_5_VLMoETextModel)

AutoConfig.register("uni_audio_rvq_qwen2_5vl_moe", UniAudioRVQQwen2_5VLMoEConfig)
AutoModelForCausalLM.register(UniAudioRVQQwen2_5VLMoEConfig, UniAudioRVQQwen2_5VLMoEForConditionalGeneration)
