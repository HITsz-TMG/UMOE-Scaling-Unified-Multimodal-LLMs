from Models.UniMoEV2 import UniMoEV2Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import deepspeed_moe_inference_utils
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
import os

model_path = "XXX/checkpoint-xxx"
resume_from_deepspeed_ckpt = model_path + "/global_stepxxx"

model = UniMoEV2Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model = model.to(device)
model = model.eval()

dist_backend = get_accelerator().communication_backend_name()
dist.init_distributed(dist_backend=dist_backend, distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, dist_init_required=True)
original_device = model.device
model = deepspeed.init_inference(  
    model,
    mp_size=1,
    ep_size=target_ep_size,
    dtype=torch_dtype, 
    replace_with_kernel_inject=False,
)
model = model.module.to(original_device)

deepspeed_ckpt = aggregation(resume_from_deepspeed_ckpt, source_ep_num=model.config.mlp_dynamic_expert_num, target_ep_size=target_ep_size, save_path=None, tie_lm_head=True)
model.load_state_dict(deepspeed_ckpt[int(os.environ["RANK"]) % target_ep_size], strict=True)
model = model.eval()

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
