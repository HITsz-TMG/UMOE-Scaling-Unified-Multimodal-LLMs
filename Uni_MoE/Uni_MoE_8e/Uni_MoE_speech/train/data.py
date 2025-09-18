

import os
import io
from dataclasses import dataclass, field
import json
import zipfile
from typing import Dict, Optional, Sequence, List, Any, Union
import random

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

import torch

import transformers

from Uni_MoE_speech.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset

from Uni_MoE_speech import conversation as conversation_lib
from Uni_MoE_speech.model import *
from Uni_MoE_speech.mm_utils import tokenizer_image_audio_video_token

from PIL import Image

AUDIOSTART = "/data/"

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def extract_image_from_zip(zip_path, image_to_extract):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(image_to_extract) as image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
    return image

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    model_eval_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    audio_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    tune_mm_audio_aligner: bool = field(default=False)
    tune_mm_audio_projector: bool = field(default=False)
    mm_audio_select_layer: Optional[int] = field(default=0)
    mm_audio_select_feature: Optional[str] = field(default="patch")
    pretrain_audio_aligner: Optional[str] = field(default=None)
    language: Optional[str] = field(default="English")
    task: Optional[str] = field(default="transcribe")
    local_files_only: Optional[str] = field(default=False)
    query_tokens_size: Optional[int] = field(default=50)
    lora_path: Optional[str] = field(default=None)
    pretrain_mlp_gate: Optional[str] = field(default=None)
    expert_dir: Optional[str] = field(default=None)
    num_experts: Optional[int] = field(default=4)
    num_experts_per_tok: Optional[int] = field(default=2)
    eval_ep_size: Optional[int] = field(default=4)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    pad_audio: bool = True
    mix_va: bool = False
    eval_data_type: str = field(default=None, metadata={"help": "Eval only. Data type of the dataset."})
    eval_output: Optional[str] = field(default=None, metadata={"help": "Eval only. Output path of the eval result."})



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    llm_lora_enable: bool = False
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mlp_dir: str = ""
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    use_pretrain_lora: bool = False # need change
    pretrain_lora_enable: bool = False # need change
    dataloader_pin_memory: bool = False # need change
    enable_deepspeed_moe: bool = False # need change

@dataclass
class EvaluateArguments(transformers.TrainingArguments):
    enable_deepspeed_moe: bool = True
    eval_ep_size: Optional[int] = 2
    vision_tower: Optional[str] = None
    audio_tower: Optional[str] = None
    model_base: Optional[str] = None
    model_path: Optional[str] = None
    version: Optional[str] = "v1"
    data_path: Optional[str] = None
    data_type: Optional[str] = None
    mlp_dir: Optional[str] = None

def preprocess_va(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = [] 
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image or has_audio or has_video:
        input_ids = torch.stack([tokenizer_image_audio_video_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image or has_audio or has_video:
                round_len = len(tokenizer_image_audio_video_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_audio_video_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False
) -> Dict:
    return preprocess_va(sources, tokenizer, has_image=has_image, has_audio=has_audio)
    
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 augment_config_path=None,
                 ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.data_path = data_path
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.mono = True
        self.sample_rate = 16000
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths_type(self):
        length_list = []
        data_type = ["video", "voice", "image"] 
        # use 3 bits to represent the modality
        # only text is 000, only image is 001, video is 010, voice is 100
        # video and voice is 110, video and image is 011, voice and image is 101
        # video, voice and image is 111
        for sample in self.list_data_dict:
            cur_len = sample["tokens"]
            cur_type = 0
            if "video" in sample:
                cur_type += 4
            if "voice" in sample:
                cur_type += 2
            if "image" in sample:
                cur_type += 1
            length_list.append((cur_len, cur_type))
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # image
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image_processor = self.data_args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if AUDIOSTART in image_file:
                image = Image.open(image_file).convert('RGB')
            else:    
                image = extract_image_from_zip(image_folder, image_file)
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # video
        if 'video' in sources[0]:
            all_frames = []
            for frame_file in self.list_data_dict[i]['video']:
                frame_folder = ""
                image_processor = self.data_args.image_processor
                # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                frame = Image.open(frame_file).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    frame = expand2square(frame, tuple(int(x*255) for x in image_processor.image_mean))
                    frame = image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0]
                else:
                    frame = image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0]
                all_frames.append(frame)
            video = torch.stack(all_frames, dim = 0)

        # deal audio
        if 'voice' in sources[0]:
            audio_processor = self.data_args.audio_processor
            audio_files = sources[0]["voice"]
            # print(audio_files)
            language = self.data_args.language
            audio_time = 20 # 30s <=> 1
            audio_len = 50
            # data input_features
            features_list = []
            features_mask = []
            for j,audio_file in enumerate(audio_files):
                sample, sample_rate = soundfile.read(audio_file, dtype='float32')
                sample = sample.T
                if self.mono:
                    sample = librosa.to_mono(sample)
                if self.sample_rate != sample_rate:
                    sample = self.resample(sample, orig_sr=sample_rate, target_sr=self.sample_rate)
                
                audio_processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
                
                tmp_sample = sample.copy()
                now_len = len(features_list)
                while len(tmp_sample) > 0:
                    # >30X16000
                    if len(tmp_sample) > 480000:
                        chunk = tmp_sample[:480001]
                        tmp_sample = tmp_sample[480001:]
                        features = audio_processor(audio=chunk, sampling_rate=self.sample_rate).input_features
                        features_list.append(features)
                        features_mask.append(j+1)
                    else:
                        # log-Mel
                        data = audio_processor(audio=tmp_sample, sampling_rate=self.sample_rate)
                        features_list.append(data["input_features"])
                        features_mask.append(j+1)
                        tmp_sample = []
                if len(features_list)-now_len>audio_time: 
                    # print("trunc") 
                    features_list = features_list[:now_len+audio_time]
                    features_mask = features_mask[:now_len+audio_time]
        # deal text
        text_len = 200
        bos_token="<s>"
        eos_token="</s>"

        data_dict = preprocess(
            [sources[0]["conversations"]],
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            has_audio=('voice' in self.list_data_dict[i]),
            has_video=('video' in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],)

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['has_image'] = True
        elif self.data_args.is_multimodal or self.data_args.mix_va:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['has_image'] = False
        if 'video' in self.list_data_dict[i]:
            data_dict['video'] = video
            data_dict['has_video'] = True
        elif self.data_args.is_multimodal or self.data_args.mix_va:
            # video does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['video'] = torch.zeros(8, 3, crop_size['height'], crop_size['width'])
            data_dict['has_video'] = False
        if 'voice' in self.list_data_dict[i]:
            data_dict["input_features"] = features_list
            data_dict["features_mask"] = features_mask
        elif self.data_args.is_multimodal or self.data_args.mix_va:
            # audio does not exist in the data, but the model is multimodal
            data_dict["input_features"] = [[np.ones((80, 3000))]]
            data_dict["features_mask"] = [0]
        
        return data_dict

    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    processor: Any

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = {}
        if "input_features" in instances[0]:
            input_features = []
            # pad features
            padlen = max([len(ins["input_features"]) for ins in instances])
            pad_features = []
            for ins in instances:
                flist = [t[0] for t in ins["input_features"]]
                while len(flist)<padlen:
                    flist.append(np.ones(flist[0].shape,dtype=flist[0].dtype)*flist[0][0,0])
                pad_features.append(flist)
            
            # pad fmask
            batch_fmask = []
            for ins in instances:
                fmask = ins["features_mask"]
                while len(fmask)<padlen:
                    fmask.append(0)
                batch_fmask.append(torch.tensor(fmask))
            # deal features
            for i in range(padlen):
                i_features = [{"input_features": feature[i]} for feature in pad_features]
                batch = self.processor.feature_extractor.pad(i_features, return_tensors="pt")
                input_features.append(batch["input_features"].unsqueeze(dim = 0).clone())
            input_feature = torch.cat(input_features, dim = 0).transpose(0,1)

            batch["input_features"] = input_feature
            batch["features_mask"] = torch.stack(batch_fmask, dim = 0)

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] =input_ids.ne(self.tokenizer.pad_token_id)

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_mask'] = torch.tensor([instance['has_image'] for instance in instances])

        if 'video' in instances[0]:
            videos = [instance['video'] for instance in instances]
            if all(x is not None and x.shape == videos[0].shape for x in videos):
                batch['videos'] = torch.stack(videos)
            else:
                batch['videos'] = videos
            batch['video_mask'] = torch.tensor([instance['has_video'] for instance in instances])

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,processor=data_args.audio_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
