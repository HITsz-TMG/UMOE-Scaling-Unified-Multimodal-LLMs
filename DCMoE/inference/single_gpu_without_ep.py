from Models.UniMoEV2 import UniMoEV2Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import deepspeed_moe_inference_utils

model_path = "XXX/checkpoint-xxx"
resume_from_deepspeed_ckpt = model_path + "/global_stepxxx"

model = UniMoEV2Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model = model.to(device)
model = model.eval()

deepspeed_ckpt = aggregation(
    resume_from_deepspeed_ckpt, 
    source_ep_num=model.config.mlp_dynamic_expert_num, 
    target_ep_size=1, 
    save_path=None, 
    tie_lm_head=False)[0]
model.load_state_dict(deepspeed_ckpt)


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
