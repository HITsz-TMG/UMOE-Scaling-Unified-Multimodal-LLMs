import os
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
import json
import logging
import pathlib
from pprint import pprint
from typing import Dict, Optional, Sequence, List, Mapping, Any
import datasets
import torch
import transformers
import re
import math
from torch.utils.data import Dataset
from PIL import Image, ImageFile

try:
    from training_utils import rank0_print
except ImportError:
    from ..training_utils import rank0_print
from .datasets_utils import debug_print, preprocess_supervised


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_IMAGE_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>{}"

SYSTEM_MESSAGE = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"""
IMPUT_FORMAT = """<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""


# copy from qwen2 vision_process


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class LazySupervisedDataset(Dataset):
    """Dataset for supervised finetuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        image_processor=None,
        image_root=None,
    ):
        super(LazySupervisedDataset, self).__init__()
        data = datasets.load_from_disk(data_path)
        data = data.shuffle(seed=233)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data = data
        self.data_args = data_args
        self.image_processor = image_processor
        self.image_root = image_root

        if hasattr(self.data_args, "data_sample") and self.data_args.data_sample is not None:
            data = data.select(range(min(self.data_args.data_sample, len(data))))
            print(f"self.data_args.data_sample is {self.data_args.data_sample}; using {len(data)} samples for training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.data[i]
        image_path = sources.get("image_path")
        has_image = image_path is not None

        if has_image:
            image_path = os.path.join(self.image_root, image_path) if self.image_root else image_path
            raw_image = Image.open(image_path).convert("RGB")

            width, height = raw_image.size

            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=IMAGE_FACTOR,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )
            image = raw_image.resize((resized_width, resized_height))

            image_inputs = self.image_processor.preprocess(
                images=image,
                videos=None,
                return_tensors="pt",
                do_resize=False,  
                do_rescale=None,  
                do_normalize=None,  
            )
            pixel_values = image_inputs["pixel_values"]
            image_grid_thw = image_inputs["image_grid_thw"]

            conversations = copy.deepcopy(sources["conversations"])
            merge_length = self.image_processor.merge_size**2
            for i in range(len(conversations)):
                sentence = conversations[i]["value"]
                if i == 0:
                    sentence = DEFAULT_IMAGE_PROMPT.format(sentence)

                while DEFAULT_IMAGE_TOKEN in sentence:
                    sentence = sentence.replace(DEFAULT_IMAGE_TOKEN, "<|placeholder|>" * (image_grid_thw.prod() // merge_length), 1)
                sentence = sentence.replace("<|placeholder|>", DEFAULT_IMAGE_TOKEN)
                conversations[i]["value"] = sentence

        else:
            conversations = copy.deepcopy(sources["conversations"])

        data_dict = preprocess_supervised(
            conversations,
            self.tokenizer,
            image_token=DEFAULT_IMAGE_TOKEN,
            image_token_index=IMAGE_TOKEN_INDEX,
            label_ignore_index=IGNORE_INDEX,
            system_message=SYSTEM_MESSAGE,
            input_format=IMPUT_FORMAT,
            has_image=has_image,
            adding_sys_in_query=False,
        )

        if has_image:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    aux_balance_weight: float

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        (
            input_ids,
            labels,
        ) = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if self.aux_balance_weight is not None:
            aux_balance_weight = torch.ones_like(input_ids)
            aux_balance_weight[labels != IGNORE_INDEX] = self.aux_balance_weight
            batch["aux_balance_weight"] = aux_balance_weight

        pixel_values = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        image_grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
        if len(pixel_values):
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        return batch
