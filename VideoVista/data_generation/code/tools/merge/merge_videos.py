# 将拆分后的视频拼接为完整视频
import os
import json
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
import subprocess
import argparse

parser = argparse.ArgumentParser(description="action recongnition")
parser.add_argument("--videos", type=str)
parser.add_argument("--timestamp", type=str)
parser.add_argument("--save", default="")
args = parser.parse_args()


base_video_path = args.videos
merge_source = json.load(open(args.timestamp, "r"))
save_path = args.save

os.makedirs(save_path, exist_ok=True)
print(len(merge_source))
results = []


def get_video_length(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps == 0:
        duration = 0
    else:
        duration = frame_count / fps
    video.release()
    return duration, fps


def concatenate_videos(video_paths, output_path):
    for video_path in video_paths:
        length, fps = get_video_length(video_path)
        print(f"{video_path}: {length} seconds  {fps} fps")

    input_cmd = ' '.join([f'-i "{path}"' for path in video_paths])

    # 构建filter_complex部分的FFmpeg命令
    filter_complex = ''.join([f'[{i}:v:0][{i}:a:0]' for i in range(len(video_paths))])
    filter_complex += f'concat=n={len(video_paths)}:v=1:a=1[outv][outa]'

    # 构建完整的FFmpeg命令
    ffmpeg_cmd = f'ffmpeg -y {input_cmd} -filter_complex "{filter_complex}" -map "[outv]" -map "[outa]" "{output_path}"'

    # 使用os.system调用FFmpeg命令
    os.system(ffmpeg_cmd)

    ffprobe_cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', output_path
    ]
    result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = float(result.stdout.strip())

    print(f"Concatenated video length: {duration} seconds")


for idx, line in enumerate(tqdm(merge_source)):
    video_name = line["video_name"]
    times = line["time"]

    all_clip_videos = []
    # time_str = ""
    for time in times:
        video_path = f"{base_video_path}/{video_name}/{video_name}.{time}.mp4"

        all_clip_videos.append(video_path)

    timestamp = args.timestamp
    timestamp = timestamp.split("_")[-1].split(".")[0]
    output_path = os.path.join(save_path, f"{video_name}.{timestamp}.{idx}.mp4")

    concatenate_videos(all_clip_videos, output_path)

