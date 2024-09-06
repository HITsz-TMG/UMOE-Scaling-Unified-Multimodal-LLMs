'''
统计切分后每个clip的视频长度
input_video_path --- 视频存放的文件夹路径
input_video_path
--- video1[.mp4]
--- video2[.mp4]

save_path --- 输出的json文件路径
json: {"video1":20,
"video2": 30,
}
'''


import os
import json
import cv2
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge")
parser.add_argument("--input_videos_path", type=str)
parser.add_argument("--save_path", type=str)
args = parser.parse_args()

base_path = args.input_videos_path
save_path = args.save_path

def get_duration_from_cv2(filename):
    # 利用视频的帧率和帧数来计算视频时长
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num / rate
        return duration


results = {}
for fileid in tqdm(os.listdir(base_path)):
    filepath = os.path.join(base_path, fileid)

    for videoname in os.listdir(filepath):
        video_path = os.path.join(filepath, videoname)

        duration = get_duration_from_cv2(video_path)

        videoid = videoname.replace(".mp4", "")

        if duration == None:
            print(videoid)

        results[videoid] = duration

json.dump(results, open(save_path, "w"), indent=2)

