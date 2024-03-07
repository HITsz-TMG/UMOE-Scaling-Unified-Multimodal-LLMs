# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
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
from PIL import Image
from io import BytesIO
import base64

import re
import torch
from transformers import StoppingCriteria
from .constants import IMAGE_TOKEN_INDEX,AUDIO_TOKEN_INDEX,VIDEO_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


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


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def tokenizer_image_audio_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):

    imgidx = [m.start() for m in re.finditer('<image>', prompt)]
    audioidx = [m.start() for m in re.finditer('<audio>', prompt)]

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in re.split(r'<image>|<audio>',prompt)]

    # print(prompt_chunks)

    def insert_separator(X, sep1, sep2, imgidx, audioidx):
        imgp = 0
        audp = 0
        all_ids = []
        for chunk in X:
            if audp>=len(audioidx) or (imgp<len(imgidx) and imgidx[imgp]<audioidx[audp]):
                sep = sep1 
                imgp += 1
            elif imgp>=len(imgidx) or (audp<len(audioidx) and imgidx[imgp]>audioidx[audp]):
                sep = sep2
                audp += 1
            all_ids.append(chunk)
            all_ids.append(sep)
        all_ids=all_ids[:-1]
        return all_ids
        # return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0  # in case that tokenizing will plus <bos>
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1),[audio_token_index] * (offset + 1), imgidx, audioidx):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def tokenizer_image_audio_video_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, video_token_index=VIDEO_TOKEN_INDEX, return_tensors=None):

    imgidx = [m.start() for m in re.finditer('<image>', prompt)]
    audioidx = [m.start() for m in re.finditer('<audio>', prompt)]
    videoidx = [m.start() for m in re.finditer('<video>', prompt)]

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in re.split(r'<image>|<audio>|<video>',prompt)]

    # print(prompt_chunks)

    def insert_separator(X, sep1, sep2, sep3, imgidx, audioidx, videoidx):
        maximal = 99999
        all_ids = []
        idxs_raw = [imgidx, audioidx, videoidx]
        seps_raw = [sep1, sep2, sep3]
        seps = []
        idxs = []
        nowp = []
        for ir,ridx in enumerate(idxs_raw):
            if len(ridx) > 0:   
                seps.append(seps_raw[ir])
                idxs.append(idxs_raw[ir])
                nowp.append(0)
        # print(X)
        if len(seps)>0:
            for chunk in X:
                nowidx = [idx[nowp[ii]] if nowp[ii]<maximal else maximal for ii,idx in enumerate(idxs)] #(imgidx[nowp[0]],audioidx[nowp[1]],videoidx[nowp[2]])
                token_id = nowidx.index(min(nowidx))
                # choose min
                sep = seps[token_id]
                # print(sep,nowidx)
                nowp[token_id]+=1 
                if nowp[token_id]>=len(idxs[token_id]):
                    # print("in")
                    nowp[token_id] = maximal

                all_ids.append(chunk)
                all_ids.append(sep)
            all_ids=all_ids[:-1]
        else:
            all_ids=X
        return all_ids
        # return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0  # in case that tokenizing will plus <bos>
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1), [audio_token_index] * (offset + 1), [video_token_index] * (offset + 1), imgidx, audioidx, videoidx):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False