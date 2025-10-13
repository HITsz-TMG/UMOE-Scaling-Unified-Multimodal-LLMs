"""
UniMoE Audio Usage Example

This example demonstrates how to use the UniMoEAudio class for:
1. Text-to-Music generation
2. Text-to-Speech (Voice Cloning)
3. Video+Text-to-Music generation
"""

import os
from utils.UniMoE_Audio_mod import UniMoEAudio

# Configuration
MODEL_PATH = "path/to/model"  # Update this to your model path
OUTPUT_DIR = "./generated_audio"
DEVICE_ID = 0

# Initialize the UniMoE Audio model
print("Initializing UniMoE Audio model...")
audio_generator = UniMoEAudio(model_path=MODEL_PATH, device_id=DEVICE_ID)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Text-to-Music Generation
print("\n=== Text-to-Music Generation ===")
music_files = audio_generator.text_to_music(
    caption="A peaceful piano melody with soft strings",
    save_name="text_music",
    output_dir=OUTPUT_DIR
)
print(f"Generated music files: {music_files}")

# 2. Text-to-Speech (Voice Cloning)
print("\n=== Text-to-Speech (Voice Cloning) ===")
speech_files = audio_generator.text_to_speech(
    caption="Hello world, this is a test of voice cloning.",
    prompt_text="They're calling to us not to give up and to keep on fighting!",
    prompt_wav="../assets/prompt_audios/en_female.wav",
    save_name="cloned_speech",
    output_dir=OUTPUT_DIR
)
print(f"Generated speech files: {speech_files}")

# 3. Video+Text-to-Music Generation
print("\n=== Video+Text-to-Music Generation ===")
video_music_files = audio_generator.video_text_to_music(
    video="../assets/audios/demo_1.mp4",
    caption="Upbeat electronic music matching the video content",
    save_name="video_music",
    output_dir=OUTPUT_DIR
)
print(f"Generated video music files: {video_music_files}")

print(f"\nAll generated files are saved in: {OUTPUT_DIR}")

