"""
读取音频信息 ：
输入：读取分割好的视频的路径 / 承接step3 /
输出：json文件 / 存储的每一个视频他的说话者 / 说话内容 / 视频id / 使用的语言lang
"""



import whisperx
import os
import json
from tqdm import tqdm
import gc
import argparse

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--input_videos_path", type=str, default="processed_data/split_videost")
parser.add_argument("--save_path", type=str, default="processed_data/meta/split_audio/audio_origin.jsonn")
args = parser.parse_args()

device = "cuda"
batch_size = 256
compute_type = "float16"
diarize_model = whisperx.DiarizationPipeline(model_name="models/speaker-diarization-3.1/config.yaml", device=device)
model = whisperx.load_model("faster-whisper-large-v2", device, compute_type=compute_type)
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
results = []

cnt = 0


def get_audio(video_path):
    global cnt
    try:
        video_id = video_path.split("/")[-1].split(".")[0] + "." + video_path.split("/")[-1].split(".")[1]
        audio = whisperx.load_audio(video_path)
        result = model.transcribe(audio, batch_size=batch_size)
        language = result["language"]
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        diarize_segments = diarize_model(audio)
        r_result = whisperx.assign_word_speakers(diarize_segments, result)
        r_result["video_id"] = video_id
        r_result["language"] = language
    except:
        cnt += 1
        r_result = {}
        r_result["video_id"] = video_id
        r_result["language"] = ""

    results.append(r_result)



l1_path = args.input_videos_path
for video_name in tqdm(os.listdir(l1_path)):
    l2_path = os.path.join(l1_path, video_name)
    for filename in os.listdir(l2_path):
        video_path = os.path.join(l2_path, filename)
        print(video_path)
        file_id = filename.split(".")[0]
        # file_id = filename.split(".")[1]
        get_audio(video_path)

    json.dump(results, open(args.save_path, "w"))

print(cnt)