import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys 
root_path = os.path.abspath("/path/to/data_preprocess") 
sys.path.append(root_path) 
from dataclasses import dataclass
from typing import Dict, Sequence, List
import json
import multiprocessing
from tqdm import tqdm
import librosa
import torch
import transformers

import conversation as conversation_lib
from mm_utils import tokenizer_image_audio_video_token
from constants import IMAGE_TOKEN_INDEX,AUDIO_TOKEN_INDEX,VIDEO_TOKEN_INDEX,IGNORE_INDEX

CACHE_PATH = "/code/easy2use/data_preprocess/testjs"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
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
    return preprocess_va(sources, tokenizer, has_image=has_image, has_audio=has_audio, has_video = has_video)

def write_json(x):
    oli,opath = x
    with open(opath, 'w') as f:
        json.dump(oli, f, indent=4)

def seperate(file_path,sep_num,cache_path):
    in_path = os.path.join(cache_path,"in")
    out_path = os.path.join(cache_path,"out")
    if os.path.exists(in_path):
        # os.rmtree(in_path)
        raise SystemError("clear the input path before seperation")
    if os.path.exists(out_path):
        # os.rmtree(out_path)
        raise SystemError("clear the output path before seperation")
    os.mkdir(in_path)
    os.mkdir(out_path)
    lines = json.load(open(file_path, 'r'))
    k, m = divmod(len(lines), sep_num)
    sep_files = [(lines[i * k + min(i, m):(i + 1) * k + min(i + 1, m)],os.path.join(in_path,f"cache_file_{i}.json")) for i in range(sep_num)]
    pool = multiprocessing.Pool()
    pool.map(write_json, sep_files)
    pool.close()
    pool.join()
    return [fs[1] for fs in sep_files]

def multiprocess_add_tokens(path):
    out_path = path.replace("/in/","/out/")
    list_data_dict = json.load(open(path, 'r'))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/code/Uni_demo_v1.5/demo_weights/llava-lora-merge-4moe-interval",
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
    all_li = []
    for i in tqdm(range(len(list_data_dict))):
        # singl = dataset[i]
        sources = list_data_dict[i]
        singl = preprocess(
            [sources["conversations"]],
            tokenizer,
            has_image=('image' in sources),
            has_audio=('voice' in sources),
            has_video=('video' in sources)
        )
        singl = singl["input_ids"][0]
        # print(sources["conversations"])
        # print(singl)
        lnt = int(singl.shape[0])
        # sources["multi_info"]["tokens"] = lnt
        # 支持多个tokens
        image_num = int(torch.sum(singl==IMAGE_TOKEN_INDEX))
        audio_num = int(torch.sum(singl==AUDIO_TOKEN_INDEX))
        video_num = int(torch.sum(singl==VIDEO_TOKEN_INDEX))
        info_dict = {}
        info_dict["tokens"] = lnt
        info_dict["image_num"] = image_num # for k in range(image_num): lnt+=575
        info_dict["video_num"] = video_num # for k in range(video_num): lnt+=575
        dur_li = []
        sep_li = []
        info_dict["audio_num"] = audio_num
        for k in range(audio_num): 
            try:
                audio, sr = librosa.load(sources["voice"][k])
                duration = librosa.get_duration(y=audio, sr=sr)
                aul = min(duration//30+1,20)
                dur_li.append(duration)
                sep_li.append(aul)
            except:
                with open("/code/easy2use/data_preprocess/error.txt","a") as f:
                    f.write(sources["voice"][k]+"\n")
            # print(duration,aul)
            # lnt+=49*aul
        info_dict["audio_dur"] = dur_li
        info_dict["audio_sep"] = sep_li
        sources["multi_info"] = info_dict
        # sources["tokens"]=lnt
        # print(i,lnt)
        all_li.append(sources)
    # print(max(all_li),min(all_li),sum(all_li)/len(all_li))
    write_json((all_li,out_path))
    
def merge_files(infiles,out_json):
    out_files = [path.replace("/in/","/out/") for path in infiles]
    all_data = []
    for d in out_files:
        with open(d,"r") as f:
            line_list = json.load(f)
            print(len(line_list))
        all_data += line_list
    print(len(all_data))
    with open(out_json,"w") as f:
        json.dump(all_data,f,indent=4) 
    

def main_process(in_json,out_json):
    infiles = seperate(in_json,1,CACHE_PATH)
    print(infiles)
    pool = multiprocessing.Pool()
    pool.map(multiprocess_add_tokens, infiles)
    pool.close()
    pool.join()
    merge_files(infiles,out_json)


if __name__ == '__main__':
    # multiprocess_add_tokens("/UNICOMFS/hitsz_mzhang_1/jsy/uni_codes/easy2use/data_preprocess/testjs/in/cache_file_0.json")
    main_process("/software/Uni_MoE_V2_jsondata/deal_bad/bad_sep.json","/software/Uni_MoE_V2_jsondata/deal_bad/bad_sep.json")
    # main_process("/UNICOMFS/hitsz_mzhang_1/jsy/multimodal_data/chart_datasets/chart_all_uni.json","/UNICOMFS/hitsz_mzhang_1/jsy/multimodal_data/chart_datasets/chart_all_uni_with_tokens.json")
        

