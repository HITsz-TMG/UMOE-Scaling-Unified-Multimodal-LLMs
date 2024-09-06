import os
import json
import shutil
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description="Splitting Move")
parser.add_argument("--input_frames_path", type=str)
parser.add_argument("--input_videos_path", type=str)
parser.add_argument("--input_splitting_points", type=str)
parser.add_argument("--output_frames_path", type=str)
parser.add_argument("--output_videos_path", type=str)
args = parser.parse_args()

input_frames_path = args.input_frames_path
output_frames_path = args.output_frames_path
input_videos_path = args.input_videos_path
output_videos_path = args.output_videos_path
splitting_infos = {}
with open(args.input_splitting_points, "r") as file:
    for line in file:
        data = json.loads(line.strip())
        splitting_infos.update(data)

os.makedirs(output_videos_path, exist_ok=True)


def seconds_to_hhmmss(seconds):
    time_delta = timedelta(seconds=seconds)
    base_time = datetime(1900, 1, 1)
    result_time = base_time + time_delta
    return result_time.strftime('%H:%M:%S')


fileids = [filename for filename in os.listdir(input_frames_path) if
           os.path.isdir(os.path.join(input_frames_path, filename))]
fileids = set(fileids)

rename_dict = {}
for fileid in tqdm(fileids):
    videopath = os.path.join(input_videos_path, f"{fileid}.mp4")
    frame_folder_path = os.path.join(input_frames_path, fileid)
    if not os.path.exists(frame_folder_path):
        print(f"Frame folder for {fileid} not found.")
        continue

    video_save_dir = os.path.join(output_videos_path, fileid)
    os.makedirs(video_save_dir, exist_ok=True)

    tar_id = 0
    oid2tid = []

    second = len(os.listdir(frame_folder_path))

    if second > 40:
        info = splitting_infos[fileid]
        begin = 0
        if info[0] == 0:
            info = info[1:]

        skip_video = True
        for tar_id, end in enumerate(info):
            tar_frames_path = os.path.join(output_frames_path, f"{fileid}.{tar_id}")
            tar_video_path = os.path.join(video_save_dir, f"{fileid}.{tar_id}.mp4")

            if not (os.path.exists(tar_frames_path) and os.path.exists(tar_video_path)):
                skip_video = False
                break

        if skip_video:
            print(f"Skipping {fileid} as it is already processed.")
            continue

        begin = 0
        tar_id = 0
        for end in info:
            tar_frames_path = os.path.join(output_frames_path, f"{fileid}.{tar_id}")
            os.makedirs(tar_frames_path, exist_ok=True)
            for base, index in enumerate(range(begin, end)):
                ori_image_path = os.path.join(frame_folder_path, f"{index}s.jpg")
                tar_image_path = os.path.join(tar_frames_path, f"{base}s.jpg")
                shutil.copy(ori_image_path, tar_image_path)

            start_time_hhmmss = seconds_to_hhmmss(begin)
            duration = end - begin

            tar_video_path = os.path.join(video_save_dir, f"{fileid}.{tar_id}.mp4")
            print(tar_video_path)
            if not os.path.exists(videopath):
                print("error")
                continue

            os.system("ffmpeg -hide_banner -loglevel panic -ss %s -t %.3f -i %s %s" % (
                start_time_hhmmss, duration, videopath, tar_video_path))

            oid2tid.append(tar_id)
            tar_id += 1
            begin = end

    else:
        tar_frames_path = os.path.join(output_frames_path, f"{fileid}.0")
        tar_video_path = os.path.join(output_videos_path, f"{fileid}")
        os.makedirs(tar_video_path, exist_ok=True)
        tar_video_path = os.path.join(tar_video_path, f"{fileid}.0.mp4")

        if os.path.exists(tar_frames_path) and os.path.exists(tar_video_path):
            print(f"Skipping {fileid} as it is already processed.")
            continue

        shutil.copy(videopath, tar_video_path)
        shutil.copytree(frame_folder_path, tar_frames_path, dirs_exist_ok=True)

        oid2tid.append(0)
        tar_id += 1

    rename_dict[fileid] = oid2tid

