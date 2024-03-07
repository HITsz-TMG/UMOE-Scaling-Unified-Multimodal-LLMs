import os 
import sys 
root_path = os.path.abspath("/path/to/Uni_MoE") 
sys.path.append(root_path) 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Any, Union
import io
import random
import librosa
import numpy as np
import soundfile
from tqdm import tqdm

import torch

import transformers

from torch.utils.data import Dataset

from Uni_MoE_speech import conversation as conversation_lib
from Uni_MoE_speech.model import *
from Uni_MoE_speech.mm_utils import tokenizer_image_audio_video_token
from Uni_MoE_speech.model.all_builder import load_all_pretrained_model
import zipfile
from PIL import Image

import argparse

AUDIOSTART = "/path/to/"


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

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

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
            if j == len(source)-1:
                conv.append_message(role, "")
            else:
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

    return dict(
        input_ids=input_ids,
        labels=None,
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
                 data_args=None,
                 augment_config_path=None,
                 ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.data_path = data_path
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        # audio sets
        self.mono = True
        self.sample_rate = 16000
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None


    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # deal with image
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
                if AUDIOSTART in frame_file:
                    frame = Image.open(frame_file).convert('RGB')
                else:    
                    frame = extract_image_from_zip(frame_folder, frame_file)
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
            if AUDIOSTART in str(sources[0]["voice"][0]):
                audio_files = sources[0]["voice"]
            else:
                audio_files = [(self.data_path[:self.data_path.rfind("/")] + li)for li in sources[0]["voice"]]
            # print(audio_files)
            language = self.data_args.language
            audio_time = 4 # 30s <=> 1
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
                while len(tmp_sample) > 0:
                    # 30X16000
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
                             labels=None,
                             data_ori= sources[0]
                             )

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
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
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
        batch["labels"] = None
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
        batch['data_ori'] = [instance['data_ori'] for instance in instances]
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

def eval(margs):
    model_path = "checkpoints/Uni-MoE-speech-e2"
    model_base = "checkpoints/Uni-MoE-speech-base"
    model_name = "unimoe_lora"
    tokenizer, model, image_processor,audio_processor, context_len = load_all_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False,vison_tower_path="checkpoints/clip-vit-large-patch14-336",audio_tower_path="checkpoints/whisper-small")
    model.bfloat16()
    model.cuda()
    model.eval()

    class args:
        data_path: str = None
        lazy_preprocess: bool = False
        is_multimodal: bool = False
        image_folder: Optional[str] = field(default=None)
        image_aspect_ratio: str = 'square'
        image_grid_pinpoints: Optional[str] = field(default=None)
        pad_audio: bool = True
        mix_va: bool = True
        def __init__(self):
            self.language = "English"
            self.task = "transcribe"
    data_args = args()

    data_args.data_path = margs.data_path

    data_args.image_processor = image_processor
    data_args.audio_processor = audio_processor

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                              data_args=data_args)
    # test data
    from torch.utils.data import DataLoader
    data = data_module["train_dataset"]
    collator = data_module["data_collator"]
    train_loader = DataLoader(data, batch_size = 1, collate_fn = collator, num_workers = 2)
    outlist = []
    print(data_args.data_path)
    if "video" in margs.data_type :
        kwargs = dict(do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024)
    if "vqa" in margs.data_type or "mmbench" in margs.data_type :
        kwargs = dict(do_sample = False,
                    num_beams = 1,
                    temperature = 0,
                    max_new_tokens=512)
    else:
        kwargs = dict(do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,)
    for step, batch in tqdm(enumerate(train_loader)):
        # print(batch['image_mask'])
        output_ids = model.generate(
                **kwargs,
                input_ids = batch["input_ids"].to(device=model.device),
                attention_mask = None,
                images = batch["images"].to(device=model.device).bfloat16() if batch["images"] is not None else None,
                image_mask = batch['image_mask'].to(device=model.device),
                videos = batch['videos'].to(device=model.device).bfloat16() if batch['videos'] is not None else None,
                video_mask = batch['video_mask'].to(device=model.device).bfloat16() if batch['video_mask'] is not None else None,
                input_features = batch["input_features"].to(device=model.device).bfloat16(), #None,
                features_mask = batch["features_mask"].to(device=model.device), #None,
            )
        outputs = tokenizer.batch_decode(output_ids[:,batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]#image要减去input_ids
        outputs = outputs.strip()
        dic = batch["data_ori"][0]
        dic["text"] = str(outputs)
        print(dic)
        outlist.append(dic)
    with open("eval/"+margs.output,"w") as f:
        json.dump(outlist, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--data_type", type=str, default=None)
    margs = parser.parse_args()
    eval(margs)
    