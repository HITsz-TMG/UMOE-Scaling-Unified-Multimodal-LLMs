# -*- coding: utf-8 -*-
import os
import sys 
root_path = os.path.abspath("/path/to/Uni_MoE") 
sys.path.append(root_path) 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import string
import flask
import json
import time
from flask import Flask,stream_with_context,Response
from flask_cors import CORS
import requests
from typing import Iterable, List
import requests
from moviepy.editor import VideoFileClip
from PIL import Image
from typing import Dict, Optional, Sequence, List, Any, Union
import librosa
import soundfile
import torch
import transformers
from Uni_MoE_speech_dp import conversation as conversation_lib
from Uni_MoE_speech_dp.model import *
from Uni_MoE_speech_dp.mm_utils import tokenizer_image_audio_video_token
from Uni_MoE_speech_dp.model.all_builder import load_all_pretrained_model
from PIL import Image


app = Flask(__name__)
CORS(app, supports_credentials=True)


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
    return preprocess_va(sources, tokenizer, has_image=has_image, has_audio=has_audio,has_video=has_video)


def initial_input(query,tokenizer, image_processor, audio_processor,image=None,voice=None,video=None):
    has_image=False
    has_audio=False
    has_video=False
    # image
    if image is not None:
        has_image=True
        image_file = image
        image = Image.open(image_file).convert('RGB')
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    # video
    if video is not None:
        has_video = True
        frames = [video+"/"+f for f in os.listdir(video)]
        all_frames = []
        for frame_file in frames:
            frame = Image.open(frame_file).convert('RGB')
            frame = image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0]
            all_frames.append(frame)
        video = torch.stack(all_frames, dim = 0)

    # audio
    language = "English"
    task = "transcribe"
    if voice is not None:
        has_audio = True
        if type(voice) == list:
            audio_files = voice
        else:
            audio_files = [voice]
        features_list = []
        padding_masks = []
        features_mask = []
        for j,audio_file in enumerate(audio_files):
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
            sample = sample.T
            sample = librosa.to_mono(sample)
            if sample_rate != 16000 :
                sample = librosa.resample(sample, orig_sr=sample_rate, target_sr=16000)
                
            audio_processor.tokenizer.set_prefix_tokens(language=language)
            
            tmp_sample = sample.copy()
            while len(tmp_sample) > 0:
                # 30X16000
                if len(tmp_sample) > 480000:
                    chunk = tmp_sample[:480001]
                    tmp_sample = tmp_sample[480001:]
                    features = audio_processor(audio=chunk, sampling_rate=16000).input_features
                    features_list.append(torch.tensor(features[0]))
                    features_mask.append(j+1)
                else:
                    # log-Mel
                    data = audio_processor(audio=tmp_sample, sampling_rate=16000)
                    features_list.append(torch.tensor(data["input_features"][0]))
                    features_mask.append(j+1)
                    tmp_sample = []

    # deal text
    if query is None: query=""
    timage = taudio = tvideo = ""
    if has_image: timage = "<image>\n"
    if has_audio: taudio = "<audio>\n"
    if has_video: tvideo = "<video>\n"
    conversations = [
        {
            "from": "human",
            "value": timage+taudio+tvideo+query
        },
        {
            "from": "gpt",
            "value": ""
        }
    ]
    data_dict = preprocess(
        [conversations],
        tokenizer,
        has_image=has_image,
        has_audio=has_audio,
        has_video=has_video
    )
    data_dict = dict(input_ids=data_dict["input_ids"][0],labels=None,data_ori= query)

    # image exist in the data
    if has_image:
        data_dict['image'] = image
        data_dict['has_image'] = True
    else:
        # image does not exist in the data, but the model is multimodal
        crop_size = image_processor.crop_size
        data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        data_dict['has_image'] = False
    if has_video:
        data_dict['video'] = video
        data_dict['has_video'] = True
    else:
        # video does not exist in the data, but the model is multimodal
        crop_size = image_processor.crop_size
        data_dict['video'] = torch.zeros(8, 3, crop_size['height'], crop_size['width'])
        data_dict['has_video'] = False
    if has_audio:
        data_dict["input_features"] =torch.stack(features_list, dim = 0)
        data_dict["features_mask"] = torch.tensor(features_mask)
    else:
        # audio does not exist in the data, but the model is multimodal
        data_dict["input_features"] = torch.stack([torch.ones((80, 3000))], dim = 0)
        data_dict["features_mask"] = torch.tensor([0])

    batch_dict = dict(
        input_ids = torch.stack([data_dict["input_ids"]], dim = 0),
        labels = None,
        attention_mask = None,
        images = torch.stack([data_dict['image']], dim = 0),
        image_mask = torch.tensor([data_dict['has_image']]),
        videos = torch.stack([data_dict['video']], dim = 0),
        video_mask = torch.tensor([data_dict['has_video']]),
        input_features = torch.stack([data_dict["input_features"]], dim = 0),
        features_mask = torch.stack([data_dict["features_mask"]], dim = 0),
    )
    
    return batch_dict

def extract_frames(video_path, output_folder, num_frames=8):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loading video files
    video_clip = VideoFileClip(video_path)

    # Calculate the number of frames and the frame interval
    total_frames = int(video_clip.duration * video_clip.fps)
    frame_interval = total_frames // num_frames

    # Extract and save frames
    frame_paths = []
    for i in range(num_frames):
        frame_time = i * frame_interval
        frame = video_clip.get_frame(frame_time / video_clip.fps)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        output_path = os.path.join(output_folder, f"frame_{i:03d}.jpg")
        pil_image.save(output_path)
        frame_paths.append(output_path)

    # Close video file
    video_clip.close()
    return output_folder

@app.route("/gen", methods=["POST"])
def progress():
    data = json.loads(flask.request.form.get('data'))
    query = data['query']
    print(data)
    path = "/path/to/tmp_files/"
    flag = data['flag']
    image_path = audio_path = frame_path = None
    if flag['image'] is not None:
        image = flask.request.files['image']
        idx = image.filename.rfind('\\') if image.filename.rfind('\\')>=0 else image.filename.rfind('/')
        image_path = path + image.filename[idx+1:]
        image.save(image_path)
    if flag['audio'] is not None:
        audio = flask.request.files['audio']
        idx = audio.filename.rfind('\\') if audio.filename.rfind('\\')>=0 else audio.filename.rfind('/')
        audio_path = path + audio.filename[idx+1:]
        audio.save(audio_path)
    if flag['video'] is not None:
        video = flask.request.files['video']
        idx = video.filename.rfind('\\') if video.filename.rfind('\\')>=0 else video.filename.rfind('/')
        video_path = path + video.filename[idx+1:]
        video.save(video_path)
        vid_aud = video_path.split(".")[0] + "_aud.mp3"
        # process video
        # audio
        try:
            clip = VideoFileClip(video_path)
            audio = clip.audio
            audio.write_audiofile(vid_aud, codec='mp3')
            audio_path=vid_aud
        except:
            pass
        # frames
        frame_folder = video_path.split(".")[0]
        frame_path = extract_frames(video_path,frame_folder)
    print(query, image_path, audio_path, frame_path)
    batch = initial_input(query=query,tokenizer=tokenizer,image_processor=image_processor,audio_processor=audio_processor,image=image_path,voice=audio_path,video=frame_path)
    if image_path is not None:
        kwargs = dict(do_sample = False,
                    num_beams = 1,
                    temperature = 0,
                    max_new_tokens=512)
    else:
        kwargs = dict(do_sample=True,
                    num_beams = 1,
                    temperature=0.2,
                    max_new_tokens=512)
    output_ids = model.generate(
            **kwargs,
            input_ids = batch["input_ids"].to(device=model.device),
            attention_mask = None,
            images = batch["images"].to(device=model.device).bfloat16() if batch["images"] is not None else None,
            image_mask = batch['image_mask'].to(device=model.device),
            videos = batch['videos'].to(device=model.device).bfloat16() if batch['videos'] is not None else None,
            video_mask = batch['video_mask'].to(device=model.device) if batch['video_mask'] is not None else None,
            input_features = batch["input_features"].to(device=model.device).bfloat16(), #None,
            features_mask = batch["features_mask"].to(device=model.device), #None,
            use_cache = True
        )
    outputs = tokenizer.batch_decode(output_ids[:,batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]#image要减去input_ids
    outputs = outputs.strip()
    print(outputs)
    return flask.jsonify({"output":outputs})



if __name__ == "__main__":
    model_path = "/path/to/Uni-MoE-speech-v1.5"
    model_base = "/path/to/Uni-MoE-speech-base-interval"
    model_name = "unimoe_lora"
    tokenizer, model, image_processor,audio_processor, context_len = load_all_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False,vison_tower_path="/data/lyx/jsy/clip-vit-large-patch14-336",audio_tower_path="/data/lyx/jsy/whisper-small")
    model.bfloat16()
    model.cuda()
    model.eval()

    app.run(host="0.0.0.0", port=9011, debug=False)  
