#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig
from Uni_MoE_speech.model.moe.modeling_llama_dp import LlamaForCausalLM, LlamaModel, LlamaAttention, LlamaFmModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.nn.functional as F

DO_GATE = True

import warnings

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

from dataclasses import dataclass


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None
    gate_info: Optional[Tuple[torch.FloatTensor]] = None


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    # print("Using flash attention")

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        max_s = q_len
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = output.view(bsz, q_len, -1)
    else:
        qkv = qkv.reshape(bsz, q_len, -1)
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask



class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaFmModel(LlavaMetaModel, LlamaFmModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaFmModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        

        if config.use_fm_block:
            self.model = LlavaLlamaFmModel(config)
        else:
            self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.use_flash_attn:
            LlamaAttention.forward = forward
            LlamaModel._prepare_decoder_attention_mask = (_prepare_decoder_attention_mask)
            LlamaFmModel._prepare_decoder_attention_mask = (_prepare_decoder_attention_mask)

        # self.init_flag = 2  # use to print in the first time
        self.post_init()
        # self.init_deepspeed_moe()  

    def get_model(self):
        return self.model

    # def init_deepspeed_moe(self):
    #     for layer_num in self.config.num_hidden_layers:
    #         if isinstance(self.model.encoder.layer[layer_num].mlp, MoE):
    #             self.model.encoder.layer[layer_num].mlp            

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        videos: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        features_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, return_indice = self.prepare_inputs_labels_for_audio_and_vision(input_ids, attention_mask, past_key_values, labels, images, image_mask, videos, video_mask, input_features, features_mask)

        if self.config.use_fm_block:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                return_indice=return_indice
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
    
        moe_loss, moe_losses = None, []
        if len(outputs[-2]) > 0:
            moe_loss_list = outputs[-2]
            for moe_loss in moe_loss_list:
                if moe_loss is not None:
                    moe_losses.append(moe_loss)

            if labels is not None and self.config.aux_balance_loss:
                # print("no aux loss!!!!!")
                # only the first to print
                # if self.init_flag >= 0:
                #     print("Using aux_balance_loss")
                #     self.init_flag -= 1
                moe_loss = self.config.aux_balance_loss_coef * sum(moe_losses)
                loss += moe_loss

        gate_info = {"indice":return_indice,"gate_logits":outputs[-1]}
        # gate_info = {"indice":return_indice,"gate_logits":None}

        if labels is not None:
            image_expert_indices = self.config.image_expert_indices
            audio_expert_indices = self.config.audio_expert_indices
            video_expert_indices = self.config.video_expert_indices
            gating_logits = outputs[-1]
            distinct_losses = []
            kl_losses = []

            for layer_idx, g_logits in enumerate(gating_logits):
                g_logits = g_logits.reshape(hidden_states.shape[0], -1, g_logits.shape[-1])
                batch_size, all_token_len, num_expert = g_logits.shape
                if g_logits.size != 0:
                    gates = F.softmax(g_logits, dim=-1)
                    indices1_s = torch.argmax(gates, dim=-1)

                    for batch_idx in range(batch_size):

                        layer_loss = 0
                        kl_loss_list = []
                        img_gates_logits = None
                        aud_gates_logits = None
                        vid_gates_logits = None
                        text_gates_logits = None

                        image_indice = return_indice["image"][batch_idx]
                        audio_indice = return_indice["audio"][batch_idx]
                        video_indice = return_indice["video"][batch_idx]
                        text_indice = return_indice["text"][batch_idx]

                        if len(image_indice) != 0:
                            img_gates = []
                            for img_ind in image_indice:
                                img_gates.append(gates[batch_idx][img_ind[0]:img_ind[1]])
                            img_gates = torch.cat(img_gates)   # (576 * image_num) * num_expert
                            img_gates_logits = torch.mean(img_gates, dim=0)
                            img_token_len = img_gates.shape[0]
                            if img_token_len > 0:
                                tp = F.one_hot(torch.tensor(image_expert_indices), num_classes=num_expert)
                                tp = torch.sum(tp, dim=0).cuda()
                                # boradcast tp to img_gates
                                tp = tp.unsqueeze(0).expand(img_token_len, -1)
                                img_loss = nn.L1Loss(reduction="mean")(img_gates, tp)
                            else: 
                                img_loss = 0
                            layer_loss += (img_token_len / all_token_len) * img_loss

                        if len(audio_indice) != 0:
                            aud_gates = []
                            for aud_ind in audio_indice:
                                aud_gates.append(gates[batch_idx][aud_ind[0]:aud_ind[1]])
                            aud_gates = torch.cat(aud_gates)
                            aud_gates_logits = torch.mean(aud_gates, dim=0)
                            aud_token_len = aud_gates.shape[0]
                            if aud_token_len > 0:
                                tp = F.one_hot(torch.tensor(audio_expert_indices), num_classes=num_expert)
                                tp = torch.sum(tp, dim=0).cuda()
                                tp = tp.unsqueeze(0).expand(aud_token_len, -1)
                                aud_loss = nn.L1Loss(reduction="mean")(aud_gates, tp)
                                # ce = F.softmax(tp, dim=0)
                                # aud_loss = torch.mean((me-ce)**2, dim=0)*num_expert*num_expert
                            else: 
                                aud_loss = 0
                            layer_loss += (aud_token_len / all_token_len) * aud_loss

                        if len(video_indice) != 0:
                            vid_gates = []
                            for vid_ind in video_indice:
                                vid_gates.append(gates[batch_idx][vid_ind[0]:vid_ind[1]])
                            vid_gates = torch.cat(vid_gates)
                            vid_gates_logits = torch.mean(vid_gates, dim=0)
                            vid_token_len = vid_gates.shape[0]
                            if vid_token_len > 0:
                                tp = F.one_hot(torch.tensor(video_expert_indices), num_classes=num_expert)
                                tp = torch.sum(tp, dim=0).cuda()
                                tp = tp.unsqueeze(0).expand(vid_token_len, -1)
                                vid_loss = nn.L1Loss(reduction="mean")(vid_gates, tp)
                            else: 
                                vid_loss = 0
                            layer_loss += (vid_token_len / all_token_len) * vid_loss

                        if len(text_indice) != 0:
                            text_gates = []
                            for text_ind in text_indice:
                                text_gates.append(gates[batch_idx][text_ind[0]:text_ind[1]])
                            text_gates = torch.cat(text_gates)
                            text_gates_logits = torch.mean(text_gates, dim=0)

                        distinct_losses.append(layer_loss)
                    
                        if img_gates_logits is not None and aud_gates_logits is not None:
                            kl_loss_list.append(1.0 - nn.KLDivLoss(reduction="batchmean")(F.log_softmax(img_gates_logits, dim=0), F.softmax(aud_gates_logits, dim=0)))
                        if img_gates_logits is not None and text_gates_logits is not None:
                            kl_loss_list.append(1.0 - nn.KLDivLoss(reduction="batchmean")(F.log_softmax(img_gates_logits, dim=0), F.softmax(text_gates_logits, dim=0)))
                        if aud_gates_logits is not None and text_gates_logits is not None:
                            kl_loss_list.append(1.0 - nn.KLDivLoss(reduction="batchmean")(F.log_softmax(aud_gates_logits, dim=0), F.softmax(text_gates_logits, dim=0)))
                        if aud_gates_logits is not None and vid_gates_logits is not None:
                            kl_loss_list.append(1.0 - nn.KLDivLoss(reduction="batchmean")(F.log_softmax(aud_gates_logits, dim=0), F.softmax(vid_gates_logits, dim=0)))
                        if vid_gates_logits is not None and text_gates_logits is not None:
                            kl_loss_list.append(1.0 - nn.KLDivLoss(reduction="batchmean")(F.log_softmax(vid_gates_logits, dim=0), F.softmax(text_gates_logits, dim=0)))

                        if len(kl_loss_list) > 0:
                            kl_losses.append(sum(kl_loss_list)/len(kl_loss_list))
                
            if labels is not None and self.config.aux_multimodal_loss:
                # if self.init_flag >= 0:
                #     print("Using aux_multimodal_loss")
                #     self.init_flag -= 1
                distinct_loss = self.config.aux_multimodal_loss_coef * sum(distinct_losses)
                loss += distinct_loss

            if labels is not None and self.config.aux_kl_loss:
                # if self.init_flag >= 0:
                #     print("Using aux_kl_loss")
                #     self.init_flag -= 1
                kl_loss = self.config.aux_kl_loss_coef * sum(kl_losses)
                loss += kl_loss
                    
        
        if not return_dict:
            outputs = outputs[:-1]+(gate_info,)
            output = (logits,) + outputs[1:]
            output = (moe_loss,) + output if moe_loss is not None else output
            return (loss,) + output if loss is not None else output
        
        # time.sleep(1)
        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_loss_list=outputs.moe_loss_list,
            gate_info=gate_info
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "input_features": kwargs.get("input_features", None),
                "features_mask": kwargs.get("features_mask", None),
                "image_mask": kwargs.get("image_mask", None),
                "video_mask": kwargs.get("video_mask", None),
                "videos": kwargs.get("videos", None),
            }
        )
        return model_inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
