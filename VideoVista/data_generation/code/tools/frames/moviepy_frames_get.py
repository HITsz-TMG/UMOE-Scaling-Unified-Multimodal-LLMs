# 抽取帧

import subprocess
import cv2
import os
import shutil
from tqdm import tqdm
# from decord import VideoReader
# from decord import cpu, gpu
from moviepy.editor import VideoFileClip
import imageio
import argparse

# 接受命令行的参数 parser
parser = argparse.ArgumentParser(description="Frames Get")
parser.add_argument("--input_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

# 得到error_file的path list / 在后续判断视频文件的时候需要用到
error_fileid_path = "/data/share/datasets/video_data/panda/error_fileid.txt"
error_file = []
with open(error_fileid_path, "r") as f:
    error_fileids = f.readlines()
    error_fileids = [x.strip() for x in error_fileids]
    for fileid in error_fileids:
        error_file.append(fileid)

# 识别save路径是否存在对应的video文件夹，如果存在，说明这个视频已经切帧好了，可以跳过处理
processed_file = []
for fileid in os.listdir(args.save_dir):
    processed_file.append(fileid)


def extract_frames(input_video, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # vr = VideoReader(input_video, ctx=cpu(0))
    # fps = int(vr.get_avg_fps())

    video = VideoFileClip(input_video)
    duration = round(video.duration)

    #这里涉及到切帧的参数设置，下面这样就是一秒一帧，0.5s采样一次 / 1.5s采样 /
    for t in range(0, duration + 1):
        # for t in np.arange(0, duration, 0.2):
        # t = round(t, 1)
        # try:
        frame = video.get_frame(t + 0.5)
        # except:
        #     frame = video.get_frame(t)
        output_path = os.path.join(output_folder, f"{t}s.jpg")
        imageio.imwrite(output_path, frame)
    video.close()


path = args.input_dir
base_save_path = args.save_dir
l1_path = path
for video_name in os.listdir(l1_path):
    # l2_path = os.path.join(l1_path, video_name)
    # for filename in tqdm(os.listdir(l2_path)):
    #     video_path = os.path.join(l2_path, filename)
    #     file_id = filename.split(".")[0] + "." + filename.split(".")[1]
    #     save_path = os.path.join(base_save_path, file_id)
    #     extract_frames(video_path, save_path)
    if video_name.endswith('.mp4') and video_name not in error_file and video_name.split(".")[0] not in processed_file:
        video_path = os.path.join(l1_path, video_name)
        file_id = video_name.split(".")[0]
        save_path = os.path.join(base_save_path, file_id)
        extract_frames(video_path, save_path)



