import copy
import os
import re

import torch
from safetensors.torch import save_file
from tqdm import tqdm

try:
    import torch_npu
except:
    pass


# MP (Zero3) is not supported yet
def aggregation(checkpoint_path, source_ep_num=8, target_ep_size=1, save_path=None, tie_lm_head=False):
    expert_pattern = r"layer_(\d+)_expert_(\d+)_mp_rank_00_model_states.pt"
    mlp_name_pattern = r"model\.layers\.(\d+)\.mlp\.dynamic_real_moe\.deepspeed_moe\.experts\.deepspeed_experts\.(\d+)"
    mlp_rename_format = "model.layers.{layer_id}.mlp.dynamic_real_moe.deepspeed_moe.experts.deepspeed_experts.{new_expert_id}{rest}"
    assert source_ep_num % target_ep_size == 0
    ep_group_num = source_ep_num // target_ep_size
    module = torch.load(os.path.join(checkpoint_path, "mp_rank_00_model_states.pt"), map_location="cpu")["module"]
    if tie_lm_head:
        module["lm_head.weight"] = module["lm_head.weight"].clone()
    target = [copy.deepcopy(module) for _ in range(target_ep_size)]
    files = os.listdir(checkpoint_path)
    for file in tqdm(files):
        match = re.match(expert_pattern, file)
        if match:
            layer_id = int(match.group(1))
            expert_id = int(match.group(2))
            param_dict = torch.load(os.path.join(checkpoint_path, file), map_location="cpu")
            for name, param in param_dict.items():
                assert name not in target[expert_id // ep_group_num]
                name_match = re.match(mlp_name_pattern, name)
                assert name_match
                assert int(name_match.group(1)) == layer_id
                assert int(name_match.group(2)) == expert_id
                rest = name[len(name_match.group(0)) :]
                new_mlp_name = mlp_rename_format.format(**{"layer_id": layer_id, "new_expert_id": expert_id % ep_group_num, "rest": rest})
                assert new_mlp_name not in target[expert_id // ep_group_num]
                target[expert_id // ep_group_num][new_mlp_name] = param

    if save_path is not None:
        for i, ckpt in enumerate(target):
            save_file(ckpt, os.path.join(save_path, f"model-expert_{i}-of-total_{len(target)}.safetensors"))

    return target

