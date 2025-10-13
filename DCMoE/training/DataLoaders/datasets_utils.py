import copy
import json
import logging
import os
import pathlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, List, Mapping, Optional, Sequence

import datasets
import tokenizers
import torch
import transformers

try:
    from training_utils import rank0_print
except ImportError:
    from ..training_utils import rank0_print


def tokenizer_image_token(
    prompt, tokenizer, 
    image_token, 
    image_token_index, 
    add_special_tokens=True
) -> List:
    prompt_chunks = [tokenizer(chunk, add_special_tokens=add_special_tokens).input_ids for chunk in prompt.split(image_token)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return input_ids


def preprocess_pretraining(
    sentence: str, 
    tokenizer: transformers.PreTrainedTokenizer, 
    image_token: str, 
    image_token_index: int, 
    label_ignore_index: int = -100, 
    has_image: bool = False, 
    truncation: bool = True
) -> Dict:
    if has_image:
        input_ids = tokenizer_image_token(
            sentence,
            tokenizer,
            image_token=image_token,
            image_token_index=image_token_index,
        )
    else:
        input_ids = tokenizer(
            [sentence],
        ).input_ids[0]

    if truncation and len(input_ids) >= tokenizer.model_max_length:
        input_ids = input_ids[: tokenizer.model_max_length]
        input_ids[-1] = tokenizer.eos_token_id
    else:
        input_ids = input_ids + [tokenizer.eos_token_id]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    eos_token_index = len(input_ids) - 1

    targets = input_ids.clone()
    if has_image:
        targets[targets == image_token_index] = label_ignore_index

    return dict(input_ids=input_ids, labels=targets, global_text_index=eos_token_index)


def preprocess_supervised(
    sentence: List[Dict],
    tokenizer: transformers.PreTrainedTokenizer,
    image_token,
    image_token_index,
    label_ignore_index,
    system_message,
    input_format,
    has_image: bool = False,
    truncation: bool = True,
    adding_sys_in_query=False,
) -> Dict:

    human_role = "human"
    ai_role = "gpt"

    assert sentence[0]["from"] == human_role

    sources = []
    targets = []
    for i, message in enumerate(sentence):
        role = message["from"]
        value = message["value"]
        assert role == ai_role if i % 2 else role == human_role
        if i == 0 and adding_sys_in_query:
            value = system_message + value 
        if i % 2 == 0:
            sources.append(input_format.format(value))
        else:
            targets.append(value + tokenizer.eos_token)

    input_ids = []
    labels = []

    if tokenizer.bos_token_id is not None:
        input_ids += [tokenizer.bos_token_id]
        labels = [label_ignore_index]

    if not adding_sys_in_query:
        input_ids = tokenizer([system_message], add_special_tokens=False).input_ids[0]
        labels = [label_ignore_index] * len(input_ids)

    for source, target in zip(sources, targets):
        if source[-1] in ["\n", "\t", " "]: 
            target = source + target.strip()
        else:
            target = source + " " + target.strip()

        if has_image:
            source_input_ids = tokenizer_image_token(
                source,
                tokenizer,
                image_token=image_token,
                image_token_index=image_token_index,
                add_special_tokens=False,
            )
            target_input_ids = tokenizer_image_token(
                target,
                tokenizer,
                image_token=image_token,
                image_token_index=image_token_index,
                add_special_tokens=False,
            )
        else:
            source_input_ids = tokenizer([source], add_special_tokens=False).input_ids[0]
            target_input_ids = tokenizer([target], add_special_tokens=False).input_ids[0]

        input_ids += target_input_ids
        labels += [label_ignore_index] * len(source_input_ids) + target_input_ids[len(source_input_ids) :]

    if truncation and len(input_ids) >= tokenizer.model_max_length:
        input_ids = input_ids[: tokenizer.model_max_length]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    if has_image:
        assert (labels[labels == image_token_index] != label_ignore_index).sum() == 0

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def debug_print(tokenizer, input_ids, labels):
    rank0_print("input_ids")
    rank0_print(input_ids)
    rank0_print("input_text")
    rank0_print(tokenizer.decode(input_ids))
    rank0_print("labels")
    rank0_print(labels)
    rank0_print("labels_text")
    labels[labels == -100] = tokenizer([" "], add_special_tokens=False).input_ids[0][0]
    rank0_print(tokenizer.decode(labels))
