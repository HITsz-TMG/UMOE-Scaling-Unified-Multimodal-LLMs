import os
import json
import argparse


def delete_files(error_file, input_folder):
    with open(error_file, 'r') as f:
        error_videos = json.load(f)
    files_in_directory = os.listdir(input_folder)

    for file_name in files_in_directory:
        # 提取文件 ID
        if file_name.endswith('.json'):
            file_id = file_name.split('.json')[0]
        else:
            file_id = file_name
        if file_id in error_videos:
            file_path = os.path.join(input_folder, file_name)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete files based on error file IDs")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing files to be deleted")
    parser.add_argument("--error_file", type=str, required=True,
                        help="Path to the JSON file containing error video IDs")

    args = parser.parse_args()

    delete_files(args.error_file, args.input)



