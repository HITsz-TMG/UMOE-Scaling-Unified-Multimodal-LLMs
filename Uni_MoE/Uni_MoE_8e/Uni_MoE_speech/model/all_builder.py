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


import os
import warnings
import shutil
import deepspeed
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from Uni_MoE_speech.model import *
from Uni_MoE_speech.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from Uni_MoE_speech.train.data import EvaluateArguments
import transformers
from Uni_MoE_speech.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from Uni_MoE_speech import conversation as conversation_lib

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def load_all_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", vison_tower_path = None, audio_tower_path = None, dp_enable = True, ep_size = 4):
    
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'unimoe' in model_name.lower():
        # Load Uni-MoE model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            lora_cfg_pretrained.ep_size = ep_size
            if vison_tower_path is not None:
                lora_cfg_pretrained.mm_vision_tower = vison_tower_path
            if audio_tower_path is not None:
                lora_cfg_pretrained.mm_audio_tower = audio_tower_path
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading Uni-MoE from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional Uni-MoE weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            if dp_enable:
                print('Loading LoRAMLP weights...')
                local_expert_num = model.config.num_experts // ep_size
                for state_dict_bias in range(model.config.ep_size):
                    state_dict_num = (state_dict_bias + dist.get_rank() * local_expert_num) % 8
                    expert_path = os.path.join(model_path, str(state_dict_num) + "_mlp_experts.bin")
                    state_dict = torch.load(expert_path, map_location=torch.device('cpu'))
                    print("Rank: ", dist.get_rank(), "Load expert from: ", expert_path, " to deepspeed_experts." + str(state_dict_bias))
                    moe_dict = {}
                    for key, value in state_dict.items():
                        # need change
                        name = "model.layers."+key.split(".")[2]+".mlp.deepspeed_moe.experts.deepspeed_experts." + str(state_dict_bias) +  "."+key.split(".")[-2]+".weight"
                        moe_dict[name] = value
                    model.load_state_dict(moe_dict, strict=False)        

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
            mm_audio_aligner_weights = torch.load(os.path.join(model_path, 'mm_audio_aligner.bin'), map_location='cpu')
            mm_audio_aligner_weights = {k: v.to(torch.float16) for k, v in mm_audio_aligner_weights.items()}
            model.load_state_dict(mm_audio_aligner_weights, strict=False)
            if os.path.exists(os.path.join(model_path, 'mlp_gates.bin')):
                gates = torch.load(os.path.join(model_path, 'mlp_gates.bin'), map_location='cpu')
                gates = {k: v.to(torch.float16) for k, v in gates.items()}
                model.load_state_dict(gates, strict=False)
        elif model_base is not None:
            # this may be mm projector only
            print('Loading Uni-MoE from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)

            mm_audio_aligner_weights = torch.load(os.path.join(model_path, 'mm_audio_aligner.bin'), map_location='cpu')
            mm_audio_aligner_weights = {k: v.to(torch.float16) for k, v in mm_audio_aligner_weights.items()}
            model.load_state_dict(mm_audio_aligner_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'unimoe' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        image_processor = None
        audio_processor = None
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=torch.float16)
            image_processor = vision_tower.image_processor

        audio_tower = model.get_audio_tower()
        if audio_tower is not None:
            if not audio_tower.is_loaded:
                audio_tower.load_model()
            audio_tower.to(device=device, dtype=torch.float16)
            audio_processor = audio_tower.audio_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if dp_enable:
        ds_engine = deepspeed.init_inference(model,
                                                # mp_size=2,
                                                # dtype=torch.half,
                                                checkpoint=None,
                                                replace_with_kernel_inject=False)
        model = ds_engine.module

    return tokenizer, model, image_processor, audio_processor, context_len

def load_all_pretrained_model_dp(model_name: str, evaluate_agrs: EvaluateArguments):
    
    global local_rank
    local_rank = evaluate_agrs.local_rank
    bnb_model_from_pretrained_args = {}

    # [1] Load model
    if evaluate_agrs.vision_tower is not None or evaluate_agrs.audio_tower is not None:
        print("[1] Load model ")
        # Load config
        pretrain_config = LlavaConfig.from_pretrained(
            evaluate_agrs.model_base,
        )
        pretrain_config.ep_size = evaluate_agrs.eval_ep_size
        pretrain_config.use_flash_attn = False

        model = LlavaLlamaForCausalLM.from_pretrained(evaluate_agrs.model_base,
                                                      config=pretrain_config,
                                                      ignore_mismatched_sizes=True,
                                                      # attn_implementation=attn_implementation,
                                                      **bnb_model_from_pretrained_args)

    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            evaluate_agrs.model_base,
            # attn_implementation=attn_implementation,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    # [2] Load expert parallel
    if evaluate_agrs.enable_deepspeed_moe:
        if evaluate_agrs.model_path is None:
            raise ValueError("model_path is required when enable_deepspeed_moe is True")
        
        print("[2] Load expert parallel")
        local_expert_num = model.config.num_experts // model.config.ep_size
        ori_expert_list = ["mlp_audiocap", "mlp_basev", "mlp_base", "mlp_base1", "mlp_base4", "mlp_base7", "mlp_base", "mlp_base7"]
        for state_dict_bias in range(local_expert_num):
            state_dict_num = (state_dict_bias + dist.get_rank() * local_expert_num) % 8
            ori_expert_path = os.path.join(evaluate_agrs.mlp_dir, ori_expert_list[state_dict_num] + ".bin")
            expert_path = os.path.join(evaluate_agrs.model_path, str(state_dict_num) + "_mlp_experts.bin")
            ori_state_dict = torch.load(ori_expert_path, map_location=torch.device('cpu'))
            state_dict = torch.load(expert_path, map_location=torch.device('cpu'))
            state_dict.update(ori_state_dict)
            print("Rank: ", dist.get_rank(), "Load expert from: ",ori_expert_path,"and", expert_path, " to deepspeed_experts." + str(state_dict_bias))

            moe_dict = {}
            for key, value in state_dict.items():
                # need change
                name = "model.layers."+key.split(".")[2]+".mlp.deepspeed_moe.experts.deepspeed_experts." + str(state_dict_bias) +  "."+key.split(".")[-2]+".weight"
                moe_dict[name] = value

            model.load_state_dict(moe_dict, strict=False)

    # [3] Load text tokenizer
    print("[3] Load text tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        evaluate_agrs.model_base,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if evaluate_agrs.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[evaluate_agrs.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # [4] Load image and audio processor
    print("[4] Load image and audio processor")
    image_processor = None
    audio_processor = None
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=evaluate_agrs.device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    audio_tower = model.get_audio_tower()
    if audio_tower is not None:
        if not audio_tower.is_loaded:
            audio_tower.load_model()
        audio_tower.to(device=evaluate_agrs.device, dtype=torch.float16)
        audio_processor = audio_tower.audio_processor
    
    # [5] Load non-lora weights (include gates) and lora weights
    if "unimoe" in model_name.lower():
        print('[5.1] Loading additional Uni-MoE weights...')
        if os.path.exists(os.path.join(evaluate_agrs.model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(evaluate_agrs.model_path, 'non_lora_trainables.bin'), map_location='cpu')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

        print('[5.2] Loading gates weights...')
        if os.path.exists(os.path.join(evaluate_agrs.model_path, 'mlp_gates.bin')):
            gates = torch.load(os.path.join(evaluate_agrs.model_path, 'mlp_gates.bin'), map_location='cpu')
            gates = {k: v.to(torch.float16) for k, v in gates.items()}
            model.load_state_dict(gates, strict=False)

        
        if os.path.exists(os.path.join(evaluate_agrs.model_path, 'fm_adapter.bin')):
            print('[5.2.5] Loading fm weights...')
            fms = torch.load(os.path.join(evaluate_agrs.model_path, 'fm_adapter.bin'), map_location='cpu')
            fms = {k: v.to(torch.float16) for k, v in fms.items()}
            model.load_state_dict(fms, strict=False)

        print('[5.3] Loading LoRA weights...')
        if "lora" in model_name:
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, evaluate_agrs.model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
            mm_audio_aligner_weights = torch.load(os.path.join(evaluate_agrs.model_path, 'mm_audio_aligner.bin'), map_location='cpu')
            mm_audio_aligner_weights = {k: v.to(torch.float16) for k, v in mm_audio_aligner_weights.items()}
            model.load_state_dict(mm_audio_aligner_weights, strict=False)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    model.requires_grad_(False)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total number of parameters: {total_num}, trained: {trainable_num}, ratio: {trainable_num/total_num:.2f}")
    

    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)

    if evaluate_agrs.enable_deepspeed_moe:
        model = deepspeed.init_inference(model,         
                                          checkpoint=None,
                                         replace_with_kernel_inject=False,)
        model = model.module

    return tokenizer, model, image_processor, audio_processor, context_len