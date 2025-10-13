import sys
import os
import torch
import torchaudio
from transformers import AutoTokenizer
import numpy as np
import gradio as gr
import tempfile
import time
import shutil
from pathlib import Path
import json
from functools import lru_cache
from loguru import logger
import threading
import itertools
import argparse
import glob
import datetime

from utils.UniMoE_Audio_release import UniMoEAudio
from examples.audio_loader import AudioPromptLoader

# Global variables
audio_model = None
audio_prompt_loader = None

# Configuration
MODEL_PATH = "path/to/your/model"
DEVICE_ID = 0

# Output directories
OUTPUT_DIR = "./gradio_outputs"
TEMP_DIR = "./gradio_outputs/temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Default configuration for TTS
DEFAULT_CONFIG = {
    "prompt_audio_storage": "./gradio_outputs/prompt_audios",
    "text_storage": "./gradio_outputs/texts",
    "prompt_audios_config": "examples/prompt_audios.json",
    "default_voice_type": "English Male",
    "default_text_length": 30,  # seconds
    "supported_languages": ["english", "chinese"],
    "supported_genders": ["male", "female"]
}

# Create default storage directories
os.makedirs(DEFAULT_CONFIG["prompt_audio_storage"], exist_ok=True)
os.makedirs(DEFAULT_CONFIG["text_storage"], exist_ok=True)

# Set temporary directory environment variables
os.environ["GRADIO_TEMP_DIR"] = TEMP_DIR
os.environ["TMPDIR"] = TEMP_DIR

# Cleanup configuration - Optimized for intensive short-term usage
CLEANUP_INTERVAL = 300   # 5 minutes - More frequent cleanup for active sessions
MAX_FILE_AGE = 900      # 1 hour - Keep files longer during intensive usage
cleanup_thread = None
cleanup_stop_event = threading.Event()



def get_reference_audio_info(language, gender):
    """Get reference audio path and text based on language and gender"""
    try:
        # Load configuration from prompt_audios.json
        config_path = "examples/prompt_audios.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        audio_prompts = config.get("audio_prompts", {})
        
        if language in audio_prompts and gender in audio_prompts[language]:
            prompt_info = audio_prompts[language][gender]
            return (prompt_info["audio_path"], prompt_info["prompt"])
        else:
            # Default fallback to English male
            default_info = audio_prompts.get("english", {}).get("male", {})
            return (default_info.get("audio_path", "assets/prompt_audios/en_male.wav"), 
                   default_info.get("prompt", "Using script blockers is generally a good idea, but it requires the user to learn a bit."))
    except Exception as e:
        logger.warning(f"Failed to load prompt_audios.json: {e}")
        # Hardcoded fallback
        audio_configs = {
            "english": {
                "male": ("assets/prompt_audios/en_male.wav", "Using script blockers is generally a good idea, but it requires the user to learn a bit."),
                "female": ("assets/prompt_audios/en_female.wav", "They're calling to us not to give up and to keep on fighting!")
            },
            "chinese": {
                "male": ("assets/prompt_audios/zh_male.wav", "小偷却一点也不气馁，继续在抽屉里翻找。"),
                "female": ("assets/prompt_audios/zh_female.wav", "然而阿卡显然已经拿定主意，要援救黑老鼠。")
            }
        }
        
        if language in audio_configs and gender in audio_configs[language]:
            return audio_configs[language][gender]
        else:
            return audio_configs["english"]["male"]

# Predefined examples
PREDEFINED_EXAMPLES = {
    "music-jazz": {
        "type": "music",
        "description": "Generate upbeat jazz music",
        "text": "A vibrant swing jazz tune featuring a walking bassline, rhythmic ride cymbals, and an improvised saxophone solo, full of fun and energy."
    },
    "music-lofi-hiphop": {
        "type": "music",
        "description": "Generate chill lo-fi hip hop beats",
        "text": "A chill lo-fi hip hop beat with a dusty vinyl crackle, mellow rhodes piano chords, a simple boom-bap drum loop, and a deep, relaxed bassline, perfect for studying or relaxing."
    },
    "voice-clone-greeting": {
        "type": "voice",
        "description": "Clone voice for friendly greeting",
        "text": "Welcome to the world of UniMoE Audio! Let's explore the infinite possibilities of AI audio together. Here, you will experience an unprecedented journey of sound creation.",
        "reference_text": get_reference_audio_info("english", "male")[1],
        "reference_audio": get_reference_audio_info("english", "male")[0]
    },
    "voice-clone-storytelling": {
        "type": "voice",
        "description": "Clone voice for storytelling",
        "text": "在群山和深海的彼岸，有一个被遗忘的魔法王国，一场伟大的冒险即将在那里展开。",
        "reference_text": get_reference_audio_info("chinese", "female")[1],
        "reference_audio": get_reference_audio_info("chinese", "female")[0]
    },
    "video-music-nature": {
        "type": "video_music",
        "description": "Generate nature-inspired music from video",
        "text": "Peaceful ambient music that captures the essence of nature, with gentle melodies and natural soundscapes."
    },
    "video-music-cinematic": {
        "type": "video_music",
        "description": "Generate cinematic background music",
        "text": "Epic cinematic orchestral music with dramatic crescendos and emotional depth, perfect for storytelling."
    }
}


def create_theme():
    """Creates a custom Gradio theme."""
    try:
        theme = gr.Theme.load("theme.json")
    except:
        theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="gray", font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"])
    return theme


def cleanup_temp_files():
    """Clean up old temporary files recursively."""
    try:
        current_time = time.time()
        files_cleaned = 0
        dirs_cleaned = 0
        
        # 递归遍历所有文件和目录
        for root, dirs, files in os.walk(TEMP_DIR):
            # 清理文件
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > MAX_FILE_AGE:
                        os.remove(file_path)
                        files_cleaned += 1
                        logger.info(f"Cleaned up old temp file: {file_path} (age: {file_age:.1f}s)")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
        
        # 清理空目录（从最深层开始）
        for root, dirs, files in os.walk(TEMP_DIR, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    # 只删除空目录
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        dirs_cleaned += 1
                        logger.info(f"Removed empty temp directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {dir_path}: {e}")
        
        if files_cleaned > 0 or dirs_cleaned > 0:
            logger.info(f"Cleanup completed: Removed {files_cleaned} old files and {dirs_cleaned} empty directories")
        else:
            logger.info("Cleanup completed: No old files or empty directories to remove")
                
    except Exception as e:
        logger.error(f"Error during temp cleanup: {e}")

def start_cleanup_thread():
    """Start the background cleanup thread."""
    global cleanup_thread
    
    def cleanup_worker():
        logger.info(f"Cleanup worker started - will run every {CLEANUP_INTERVAL} seconds, removing files older than {MAX_FILE_AGE} seconds")
        while not cleanup_stop_event.is_set():
            cleanup_temp_files()
            # Wait for the specified interval or until stop event is set
            if cleanup_stop_event.wait(CLEANUP_INTERVAL):
                break  # Stop event was set, exit the loop
        logger.info("Cleanup worker stopped")
    
    if cleanup_thread is None or not cleanup_thread.is_alive():
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info(f"Cleanup thread started with interval={CLEANUP_INTERVAL}s, max_age={MAX_FILE_AGE}s")

def stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global cleanup_thread
    cleanup_stop_event.set()
    if cleanup_thread and cleanup_thread.is_alive():
        cleanup_thread.join(timeout=5)
        logger.info("Stopped temp cleanup thread")

def initialize_model():
    """Initialize UniMoE Audio model."""
    global audio_model, audio_prompt_loader
    if audio_model is not None: return
    if not torch.cuda.is_available(): raise RuntimeError("This application requires an NVIDIA GPU and CUDA environment.")
    logger.info("Initializing UniMoE Audio model...")
    audio_model = UniMoEAudio(MODEL_PATH, device_id=DEVICE_ID)
    logger.info("Model initialization complete!")
    
    # Initialize audio prompt loader
    logger.info("Initializing Audio Prompt Loader...")
    audio_prompt_loader = AudioPromptLoader()
    logger.info("Audio Prompt Loader initialization complete!")
    
    # Start cleanup thread after model initialization
    start_cleanup_thread()



def generate_music(caption, cfg_scale=10.0, temperature=1.0, max_audio_seconds=10,
                   top_p=1.0, cfg_filter_top_k=45, eos_prob_mul_factor=1.0, do_sample=True):
    """Music generation function with animation."""
    try:
        if not caption.strip():
            yield gr.update(), "Please enter a music description."
            return
        
        yield gr.update(), "Generating music...Please kindly wait"
        start_time = time.time()
        
        # Use UniMoEAudio class for generation
        output_paths = audio_model.text_to_music(
            caption=caption,
            output_dir=OUTPUT_DIR,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            eos_prob_mul_factor=eos_prob_mul_factor,
            do_sample=do_sample
        )
        
        generation_time = time.time() - start_time
        
        if output_paths and len(output_paths) > 0:
            output_path = output_paths[0]  # Use the first generated file
            success_msg = f"Music generation successful!\nTime taken: {generation_time:.2f}s\nFile: {os.path.basename(output_path)}"
            yield output_path, success_msg
        else:
            yield gr.update(), "Music generation failed."
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        yield gr.update(), f"Generation failed:\n{str(e)}"

def generate_voice_clone(target_text, reference_audio, reference_text, cfg_scale=1.0,
                         temperature=1.0, max_audio_seconds=30, top_p=1.0,
                         cfg_filter_top_k=45, eos_prob_mul_factor=1.0, do_sample=True):
    """Voice cloning function with animation."""
    try:
        if not target_text.strip():
            yield gr.update(), "Please enter the target text."
            return
        if reference_audio is None:
            yield gr.update(), "Please upload a reference audio file."
            return
        if not reference_text.strip():
            yield gr.update(), "Please enter the reference audio transcript."
            return

        yield gr.update(), "Starting voice cloning process... Please kindly wait"
        start_time = time.time()
        
        # Use UniMoEAudio class for generation
        output_paths = audio_model.text_to_speech(
            caption=target_text,
            prompt_text=reference_text,
            prompt_wav=reference_audio,
            output_dir=OUTPUT_DIR,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            eos_prob_mul_factor=eos_prob_mul_factor,
            do_sample=do_sample
        )
        
        generation_time = time.time() - start_time
        
        if output_paths and len(output_paths) > 0:
            output_path = output_paths[0]  # Use the first generated file
            success_msg = f"Voice cloning successful!\nTime taken: {generation_time:.2f}s\nFile: {os.path.basename(output_path)}"
            yield output_path, success_msg
        else:
            yield gr.update(), "Voice cloning failed."

    except Exception as e:
        logger.error(f"Cloning error: {str(e)}")
        yield gr.update(), f"Cloning failed:\n{str(e)}"

def generate_tts(target_text, voice_type, cfg_scale=1.0, temperature=1.0, 
                 max_audio_seconds=30, top_p=1.0, cfg_filter_top_k=45, 
                 eos_prob_mul_factor=1.0, do_sample=True):
    """Text-to-Speech generation function using predefined voice types."""
    try:
        if not target_text.strip():
            yield gr.update(), "Please enter the target text."
            return

        yield gr.update(), "Starting TTS generation... Please kindly wait"
        start_time = time.time()
        
        # Map voice type to audio info
        voice_mapping = {
            "English Male": ("english", "male"),
            "English Female": ("english", "female"), 
            "Chinese Male": ("chinese", "male"),
            "Chinese Female": ("chinese", "female")
        }
        
        if voice_type not in voice_mapping:
            yield gr.update(), f"Unsupported voice type: {voice_type}"
            return
            
        language, gender = voice_mapping[voice_type]
        reference_audio, reference_text = get_reference_audio_info(language, gender)
        
        if reference_audio is None:
            yield gr.update(), f"Failed to load reference audio for {voice_type}"
            return
        
        # Save input text to default storage location for future reference
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            text_filename = f"tts_input_{timestamp}.txt"
            text_path = os.path.join(DEFAULT_CONFIG["text_storage"], text_filename)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Voice Type: {voice_type}\n")
                f.write(f"Generated Time: {timestamp}\n")
                f.write(f"Text: {target_text}\n")
                f.write(f"Reference Audio: {reference_audio}\n")
                f.write(f"Reference Text: {reference_text}\n")
        except Exception as e:
            logger.warning(f"Failed to save input text: {e}")
        
        # Use UniMoEAudio class for generation
        output_paths = audio_model.text_to_speech(
            caption=target_text,
            prompt_text=reference_text,
            prompt_wav=reference_audio,
            output_dir=OUTPUT_DIR,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            eos_prob_mul_factor=eos_prob_mul_factor,
            do_sample=do_sample
        )
        
        generation_time = time.time() - start_time
        
        if output_paths and len(output_paths) > 0:
            output_path = output_paths[0]  # Use the first generated file
            success_msg = f"TTS generation successful!\nVoice: {voice_type}\nTime taken: {generation_time:.2f}s\nFile: {os.path.basename(output_path)}"
            yield output_path, success_msg
        else:
            yield gr.update(), "TTS generation failed."

    except Exception as e:
        logger.error(f"TTS generation error: {str(e)}")
        yield gr.update(), f"TTS generation failed:\n{str(e)}"

def generate_video_music(video_file, caption, cfg_scale=10.0, temperature=1.0, 
                        top_p=1.0, cfg_filter_top_k=45, eos_prob_mul_factor=0.6, 
                        do_sample=True, fps=1, max_frames=1):
    """Video+Text to music generation function."""
    try:
        if not caption.strip():
            yield gr.update(), "Please enter a music description."
            return
        if video_file is None:
            yield gr.update(), "Please upload a video file."
            return

        yield gr.update(), "Generating music from video and text...Please kindly wait"
        start_time = time.time()
        
        # Use UniMoEAudio class for generation
        output_paths = audio_model.video_text_to_music(
            video=video_file,
            caption=caption,
            output_dir=OUTPUT_DIR,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            eos_prob_mul_factor=eos_prob_mul_factor,
            do_sample=do_sample,
            fps=fps,
            max_frames=max_frames
        )
        
        generation_time = time.time() - start_time
        
        if output_paths and len(output_paths) > 0:
            output_path = output_paths[0]  # Use the first generated file
            success_msg = f"Video music generation successful!\nTime taken: {generation_time:.2f}s\nFile: {os.path.basename(output_path)}"
            yield output_path, success_msg
        else:
            yield gr.update(), "Video music generation failed."
        
    except Exception as e:
        logger.error(f"Video music generation error: {str(e)}")
        yield gr.update(), f"Video music generation failed:\n{str(e)}"

def create_demo():
    """Create the Gradio demo interface with enforced left/right layout."""
    logger.info("Initializing model...")
    initialize_model()
    theme = create_theme()

    # CSS for left/right layout
    enhanced_css = """
    /* Force left/right split layout */
    .main-row {
        display: flex !important;
        flex-direction: row !important;
        gap: 30px !important;
        align-items: flex-start !important;
    }
    
    .left-column {
        flex: 3 !important;
        min-width: 600px !important;
        max-width: 800px !important;
        /* border-right: 3px solid #e0e0e0 !important; */
        padding-right: 20px !important;
    }
    
    .right-column {
        flex: 2 !important;
        min-width: 400px !important;
        background: #f8f9fa !important;
        padding: 25px !important;
        border-radius: 15px !important;
        border: 1px solid #e0e0e0 !important;
        position: sticky !important;
        top: 20px !important;
    }
    
    /* Main container style */
    .gradio-container { 
        max-width: 1600px !important; 
        margin: auto !important; 
        padding: 20px !important; 
    }
    
    /* Title style */
    .main-title { 
        text-align: center; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        background-clip: text; 
        font-size: 2.5em !important; 
        font-weight: bold !important; 
        margin-bottom: 0.5em !important; 
    }
    
    .subtitle { 
        text-align: center; 
        color: var(--body-text-color-subdued); 
        font-size: 1.2em; 
        margin-bottom: 2em; 
    }
    
    /* Button styles */
    .primary-button { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
        border: none !important; 
        color: white !important; 
        font-weight: bold !important; 
        border-radius: 8px !important; 
        padding: 12px 24px !important;
    }
    
    .secondary-button { 
        background: var(--background-fill-secondary) !important; 
        border: 1px solid var(--border-color-primary) !important; 
        color: var(--body-text-color) !important; 
        font-weight: 500 !important; 
        border-radius: 8px !important; 
    }
    
    /* Example row style */
    .example-row { 
        padding: 15px !important; 
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
        background: var(--background-fill-primary) !important;
    }
    
    /* Component spacing optimization */
    .input-group {
        margin-bottom: 20px !important;
        padding: 20px !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        background: white !important;
    }
    
    .output-section {
        background: white !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin-bottom: 15px !important;
        border: 1px solid #e0e0e0 !important;
    }

    .download-file-group .wrapper {
        padding-top: 5px !important;
        padding-bottom: 5px !important;
    }
    .download-file-group {
        padding-top: 5px !important;
        padding-bottom: 5px !important;
    }
    
    /* Mobile adaptation */
    @media (max-width: 1024px) { 
        .main-row {
            flex-direction: column !important;
        }
        
        .left-column {
            border-right: none !important;
            padding-right: 0 !important;
            margin-bottom: 20px !important;
            min-width: auto !important;
        }
        
        .right-column {
            position: static !important;
        }
    }
    """

    with gr.Blocks(css=enhanced_css, theme=theme, title="UniMoE Audio Studio") as demo:
        gr.HTML('<h1 class="main-title">UniMoE Audio Studio</h1>')
        # Main layout with forced left/right split
        with gr.Row(elem_classes=["main-row"]):
            
            with gr.Column(elem_classes=["left-column"]):
                # gr.HTML('<div style="border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-bottom: 20px;"><h3>Input Controls</h3></div>')
                gr.HTML('<div style="border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-bottom: 20px;"><h3>Input Controls</h3></div>')
                
                mode_switch = gr.Radio(
                    ["Voice Cloning", "Text-to-Speech (TTS)", "Music Generation", "Video+Text to Music"],
                    label="Select Mode",
                    value="Voice Cloning",
                    container=True
                )

                with gr.Group(elem_classes=["input-group"]):
                    gr.Markdown("### Generation Inputs")
                    
                    main_text_input = gr.Textbox(
                        label="Target Text to Generate", 
                        placeholder="Enter the text you want the cloned voice to speak...", 
                        lines=3
                    )
                    
                    reference_audio_input = gr.Audio(
                        label="Reference Audio (Upload voice sample)", 
                        type="filepath",
                    )
                    
                    reference_text_input = gr.Textbox(
                        label="Reference Audio Transcript", 
                        placeholder="Enter exactly what is said in the reference audio...", 
                        lines=2
                    )
                    
                    # TTS Voice Selection (hidden by default)
                    tts_voice_selection = gr.Radio(
                        choices=["English Male", "English Female", "Chinese Male", "Chinese Female"],
                        label="Select Voice Type",
                        value="English Male",
                        visible=False
                    )
                    
                    video_input = gr.Video(
                        label="Reference Video (Upload video file)",
                        visible=False
                    )

                with gr.Accordion("Advanced Settings (Voice Cloning)", open=False, visible=True) as voice_accordion:
                    with gr.Row():
                        voice_cfg_scale = gr.Slider(minimum=0.5, maximum=5.0, value=1.0, step=0.1, label="CFG Scale")
                        voice_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                    with gr.Row():
                        voice_max_seconds = gr.Slider(minimum=10, maximum=60, value=30, step=5, label="Max Audio Length (s)")
                        voice_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p Sampling")
                    with gr.Row():
                        voice_cfg_filter_top_k = gr.Slider(minimum=10, maximum=100, value=45, step=5, label="CFG Filter Top-k")
                        voice_eos_prob_mul = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="EOS Probability Factor")
                    voice_do_sample = gr.Checkbox(value=True, label="Enable Sampling")

                with gr.Accordion("Advanced Settings (TTS)", open=False, visible=False) as tts_accordion:
                    with gr.Row():
                        tts_cfg_scale = gr.Slider(minimum=0.5, maximum=5.0, value=1.0, step=0.1, label="CFG Scale")
                        tts_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                    with gr.Row():
                        tts_max_seconds = gr.Slider(minimum=10, maximum=60, value=30, step=5, label="Max Audio Length (s)")
                        tts_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p Sampling")
                    with gr.Row():
                        tts_cfg_filter_top_k = gr.Slider(minimum=10, maximum=100, value=45, step=5, label="CFG Filter Top-k")
                        tts_eos_prob_mul = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="EOS Probability Factor")
                    tts_do_sample = gr.Checkbox(value=True, label="Enable Sampling")

                with gr.Accordion("Advanced Settings (Music Generation)", open=False, visible=False) as music_accordion:
                    with gr.Row():
                        music_cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, value=10.0, step=0.5, label="CFG Scale")
                        music_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                    with gr.Row():
                        music_max_seconds = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Max Audio Length (s)")
                        music_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p Sampling")
                    with gr.Row():
                        music_cfg_filter_top_k = gr.Slider(minimum=10, maximum=100, value=45, step=5, label="CFG Filter Top-k")
                        music_eos_prob_mul = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="EOS Probability Factor")
                    music_do_sample = gr.Checkbox(value=True, label="Enable Sampling")

                with gr.Accordion("Advanced Settings (Video+Text to Music)", open=False, visible=False) as video_music_accordion:
                    with gr.Row():
                        video_music_cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, value=10.0, step=0.5, label="CFG Scale")
                        video_music_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                    with gr.Row():
                        video_music_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p Sampling")
                        video_music_cfg_filter_top_k = gr.Slider(minimum=10, maximum=100, value=45, step=5, label="CFG Filter Top-k")
                    with gr.Row():
                        video_music_eos_prob_mul = gr.Slider(minimum=0.5, maximum=2.0, value=0.6, step=0.1, label="EOS Probability Factor")
                        video_music_fps = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Video FPS")
                    with gr.Row():
                        video_music_max_frames = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Max Frames")
                        video_music_do_sample = gr.Checkbox(value=True, label="Enable Sampling")

                with gr.Row():
                    clear_btn = gr.Button("Clear All", elem_classes=["secondary-button"])
                    generate_btn = gr.Button("Generate", variant="primary", elem_classes=["primary-button"])

                with gr.Group(visible=True) as vc_examples_group:
                    gr.Markdown("### Example Templates (Voice Cloning)")
                    voice_examples_buttons = []
                    for key, example in PREDEFINED_EXAMPLES.items():
                        if example["type"] == "voice":
                            with gr.Row(variant="panel", elem_classes=["example-row"]):
                                with gr.Column(scale=4): 
                                    gr.Markdown(f"**{key.replace('-', ' ').title()}**\n\n*{example['description']}*")
                                with gr.Column(scale=1, min_width=80):
                                    btn = gr.Button("Use", size="sm", elem_classes=["secondary-button"])
                                    voice_examples_buttons.append((btn, key, example))

                with gr.Group(visible=False) as tts_examples_group:
                    gr.Markdown("### Example Templates (Text-to-Speech)")
                    tts_examples_buttons = []
                    tts_examples = [
                        ("English Male Greeting", "Hello! Welcome to the world of AI-generated speech. This is a demonstration of our text-to-speech technology.", "English Male"),
                        ("English Female Announcement", "Attention passengers, the next train will arrive at platform 2 in approximately 5 minutes.", "English Female"),
                        ("Chinese Male Story", "在遥远的古代，有一位智慧的老人，他知道许多神奇的故事。", "Chinese Male"),
                        ("Chinese Female Poetry", "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。", "Chinese Female")
                    ]
                    
                    for title, text, voice in tts_examples:
                        with gr.Row(variant="panel", elem_classes=["example-row"]):
                            with gr.Column(scale=4):
                                gr.Markdown(f"**{title}**\n\n*Voice: {voice}*\n\n{text[:50]}...")
                            with gr.Column(scale=1, min_width=80):
                                btn = gr.Button("Use", size="sm", elem_classes=["secondary-button"])
                                tts_examples_buttons.append((btn, title, text, voice))

                with gr.Group(visible=False) as music_examples_group:
                    gr.Markdown("### Example Templates (Music Generation)")
                    music_examples_buttons = []
                    for key, example in PREDEFINED_EXAMPLES.items():
                        if example["type"] == "music":
                            with gr.Row(variant="panel", elem_classes=["example-row"]):
                                with gr.Column(scale=4): 
                                    gr.Markdown(f"**{key.replace('-', ' ').title()}**\n\n*{example['description']}*")
                                with gr.Column(scale=1, min_width=80):
                                    btn = gr.Button("Use", size="sm", elem_classes=["secondary-button"])
                                    music_examples_buttons.append((btn, key, example))

                with gr.Group(visible=False) as video_music_examples_group:
                    gr.Markdown("### Example Templates (Video+Text to Music)")
                    video_music_examples_buttons = []
                    for key, example in PREDEFINED_EXAMPLES.items():
                        if example["type"] == "video-music":
                            with gr.Row(variant="panel", elem_classes=["example-row"]):
                                with gr.Column(scale=4): 
                                    gr.Markdown(f"**{key.replace('-', ' ').title()}**\n\n*{example['description']}*")
                                with gr.Column(scale=1, min_width=80):
                                    btn = gr.Button("Use", size="sm", elem_classes=["secondary-button"])
                                    video_music_examples_buttons.append((btn, key, example))
            
            with gr.Column(elem_classes=["right-column"]):
                gr.HTML('<div style="border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-bottom: 20px;"><h3>Output Results</h3></div>')
                
                with gr.Group(elem_classes=["output-section"]):
                    output_audio = gr.Audio(
                        label="Generated Audio", 
                        interactive=False, 
                        autoplay=False
                    )
                
                # with gr.Group(elem_classes=["output-section", "download-file-group"]):
                #     output_download = gr.File(
                #         label="Download File", 
                #         interactive=False
                #     )
                
                with gr.Group(elem_classes=["output-section"]):
                    status_textbox = gr.Textbox(
                        label="Generation Status", 
                        interactive=False, 
                        lines=8, 
                        value="Ready to clone voice..."
                    )

        
        def update_ui_for_mode(mode):
            if mode == "Voice Cloning":
                return {
                    main_text_input: gr.update(value="", label="Target Text to Generate", placeholder="Enter the text you want the cloned voice to speak..."),
                    reference_audio_input: gr.update(value=None, interactive=True, visible=True),
                    reference_text_input: gr.update(value="", interactive=True, visible=True, placeholder="Enter exactly what is said in the reference audio..."),
                    tts_voice_selection: gr.update(visible=False),
                    video_input: gr.update(value=None, visible=False),
                    voice_accordion: gr.update(visible=True),
                    tts_accordion: gr.update(visible=False),
                    music_accordion: gr.update(visible=False),
                    video_music_accordion: gr.update(visible=False),
                    vc_examples_group: gr.update(visible=True),
                    tts_examples_group: gr.update(visible=False),
                    music_examples_group: gr.update(visible=False),
                    video_music_examples_group: gr.update(visible=False),
                    output_audio: gr.update(value=None),
                    status_textbox: gr.update(value="Ready to clone voice..."),
                }
            elif mode == "Text-to-Speech (TTS)":
                return {
                    main_text_input: gr.update(value="", label="Text to Speak", placeholder="Enter the text you want to convert to speech..."),
                    reference_audio_input: gr.update(value=None, interactive=False, visible=False),
                    reference_text_input: gr.update(value="", interactive=False, visible=False),
                    tts_voice_selection: gr.update(visible=True),
                    video_input: gr.update(value=None, visible=False),
                    voice_accordion: gr.update(visible=False),
                    tts_accordion: gr.update(visible=True),
                    music_accordion: gr.update(visible=False),
                    video_music_accordion: gr.update(visible=False),
                    vc_examples_group: gr.update(visible=False),
                    tts_examples_group: gr.update(visible=True),
                    music_examples_group: gr.update(visible=False),
                    video_music_examples_group: gr.update(visible=False),
                    output_audio: gr.update(value=None),
                    status_textbox: gr.update(value="Ready to generate speech..."),
                }
            elif mode == "Music Generation":
                return {
                    main_text_input: gr.update(value="", label="Music Description", placeholder="e.g., A vibrant swing jazz tune featuring a walking bassline..."),
                    reference_audio_input: gr.update(value=None, interactive=False, visible=True),
                    reference_text_input: gr.update(value="", interactive=False, visible=True, placeholder="-- Not used for Music Generation --"),
                    tts_voice_selection: gr.update(visible=False),
                    video_input: gr.update(value=None, visible=False),
                    voice_accordion: gr.update(visible=False),
                    tts_accordion: gr.update(visible=False),
                    music_accordion: gr.update(visible=True),
                    video_music_accordion: gr.update(visible=False),
                    vc_examples_group: gr.update(visible=False),
                    tts_examples_group: gr.update(visible=False),
                    music_examples_group: gr.update(visible=True),
                    video_music_examples_group: gr.update(visible=False),
                    output_audio: gr.update(value=None),
                    status_textbox: gr.update(value="Ready to generate music..."),
                }
            else: # Video+Text to Music
                return {
                    main_text_input: gr.update(value="", label="Music Description", placeholder="e.g., A cinematic orchestral piece with dramatic crescendos..."),
                    reference_audio_input: gr.update(value=None, visible=False),
                    reference_text_input: gr.update(value="", visible=False),
                    tts_voice_selection: gr.update(visible=False),
                    video_input: gr.update(value=None, visible=True),
                    voice_accordion: gr.update(visible=False),
                    tts_accordion: gr.update(visible=False),
                    music_accordion: gr.update(visible=False),
                    video_music_accordion: gr.update(visible=True),
                    vc_examples_group: gr.update(visible=False),
                    tts_examples_group: gr.update(visible=False),
                    music_examples_group: gr.update(visible=False),
                    video_music_examples_group: gr.update(visible=True),
                    output_audio: gr.update(value=None),
                    status_textbox: gr.update(value="Ready to generate video-based music..."),
                }
        
        mode_switch.change(
            fn=update_ui_for_mode,
            inputs=mode_switch,
            outputs=[
                main_text_input, reference_audio_input, reference_text_input, tts_voice_selection, video_input,
                voice_accordion, tts_accordion, music_accordion, video_music_accordion,
                vc_examples_group, tts_examples_group, music_examples_group, video_music_examples_group,
                output_audio, status_textbox
            ]
        )

        def clear_all_inputs():
            return ("", None, "", "English Male", None, None, "Ready...")

        clear_btn.click(
            fn=clear_all_inputs, 
            outputs=[main_text_input, reference_audio_input, reference_text_input, tts_voice_selection, video_input, output_audio, status_textbox]
            # outputs=[main_text_input, reference_audio_input, reference_text_input, tts_voice_selection, video_input, output_audio, output_download, status_textbox]
        )
        
        def on_generate_click(mode, main_text, ref_audio, ref_text, tts_voice, video_file,
                              vc_cfg, vc_temp, vc_sec, vc_p, vc_k, vc_eos, vc_sample,
                              tts_cfg, tts_temp, tts_sec, tts_p, tts_k, tts_eos, tts_sample,
                              m_cfg, m_temp, m_sec, m_p, m_k, m_eos, m_sample,
                              vm_cfg, vm_temp, vm_p, vm_k, vm_eos, vm_fps, vm_frames, vm_sample):
            if mode == "Voice Cloning":
                yield from generate_voice_clone(main_text, ref_audio, ref_text, vc_cfg, vc_temp, vc_sec, vc_p, vc_k, vc_eos, vc_sample)
            elif mode == "Text-to-Speech (TTS)":
                yield from generate_tts(main_text, tts_voice, tts_cfg, tts_temp, tts_sec, tts_p, tts_k, tts_eos, tts_sample)
            elif mode == "Music Generation":
                yield from generate_music(main_text, m_cfg, m_temp, m_sec, m_p, m_k, m_eos, m_sample)
            else: # Video+Text to Music
                yield from generate_video_music(video_file, main_text, vm_cfg, vm_temp, vm_p, vm_k, vm_eos, vm_sample, vm_fps, vm_frames)

        generate_btn.click(
            fn=on_generate_click, 
            inputs=[
                mode_switch, main_text_input, reference_audio_input, reference_text_input, tts_voice_selection, video_input,
                voice_cfg_scale, voice_temperature, voice_max_seconds, voice_top_p, voice_cfg_filter_top_k, voice_eos_prob_mul, voice_do_sample,
                tts_cfg_scale, tts_temperature, tts_max_seconds, tts_top_p, tts_cfg_filter_top_k, tts_eos_prob_mul, tts_do_sample,
                music_cfg_scale, music_temperature, music_max_seconds, music_top_p, music_cfg_filter_top_k, music_eos_prob_mul, music_do_sample,
                video_music_cfg_scale, video_music_temperature, video_music_top_p, video_music_cfg_filter_top_k, video_music_eos_prob_mul, video_music_fps, video_music_max_frames, video_music_do_sample
            ], 
            outputs=[output_audio, status_textbox], 
            # outputs=[output_audio, output_download, status_textbox], 
            show_progress="hidden"
        )
        
        def load_voice_example(text, ref_text, ref_audio, key):
             return {
                 main_text_input: gr.update(value=text),
                 reference_text_input: gr.update(value=ref_text),
                 reference_audio_input: gr.update(value=ref_audio),
                 output_audio: gr.update(value=None),
                #  output_download: gr.update(value=None),
                 status_textbox: gr.update(value=f"Template loaded: {key.replace('-', ' ').title()}")
             }

        def load_music_example(text, key):
            return {
                main_text_input: gr.update(value=text),
                output_audio: gr.update(value=None),
                # output_download: gr.update(value=None),
                status_textbox: gr.update(value=f"Template loaded: {key.replace('-', ' ').title()}")
            }

        def load_tts_example(text, voice_type, key):
            return {
                main_text_input: gr.update(value=text),
                tts_voice_selection: gr.update(value=voice_type),
                output_audio: gr.update(value=None),
                status_textbox: gr.update(value=f"TTS Template loaded: {key.replace('-', ' ').title()}")
            }

        for btn, key, example in voice_examples_buttons:
            btn.click(
                fn=load_voice_example, 
                inputs=[
                    gr.Textbox(value=example["text"], visible=False), 
                    gr.Textbox(value=example.get("reference_text", ""), visible=False),
                    gr.Textbox(value=example.get("reference_audio", ""), visible=False),
                    gr.Textbox(value=key, visible=False)
                ], 
                # outputs=[main_text_input, reference_text_input, reference_audio_input, output_audio, output_download, status_textbox]
                outputs=[main_text_input, reference_text_input, reference_audio_input, output_audio, status_textbox]

            )
        
        for btn, key, example in music_examples_buttons:
            btn.click(
                fn=load_music_example, 
                inputs=[gr.Textbox(value=example["text"], visible=False), gr.Textbox(value=key, visible=False)], 
                # outputs=[main_text_input, output_audio, output_download, status_textbox]
                outputs=[main_text_input, output_audio, status_textbox]

            )

        for btn, key, text, voice in tts_examples_buttons:
            btn.click(
                fn=load_tts_example, 
                inputs=[
                    gr.Textbox(value=text, visible=False), 
                    gr.Textbox(value=voice, visible=False),
                    gr.Textbox(value=key, visible=False)
                ], 
                outputs=[main_text_input, tts_voice_selection, output_audio, status_textbox]
            )

        def load_video_music_example(text, key):
            return {
                main_text_input: gr.update(value=text),
                output_audio: gr.update(value=None),
                # output_download: gr.update(value=None),
                status_textbox: gr.update(value=f"Template loaded: {key.replace('-', ' ').title()}")
            }

        for btn, key, example in video_music_examples_buttons:
            btn.click(
                fn=load_video_music_example, 
                inputs=[gr.Textbox(value=example["text"], visible=False), gr.Textbox(value=key, visible=False)], 
                # outputs=[main_text_input, output_audio, output_download, status_textbox]
                outputs=[main_text_input, output_audio, status_textbox]

            )

    return demo

def main():
    """Main function to parse arguments and launch the demo."""
    global MODEL_PATH, DEVICE_ID
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="UniMoE Audio Web Demo")
    parser.add_argument("--model", type=str, default= MODEL_PATH, 
                        help="Path to the model directory (default: ./models/UniMoE-Audio-preview)")
    parser.add_argument("--device", type=int, default=0, 
                        help="CUDA device ID (default: 0)")
    parser.add_argument("--port", type=int, default=7860, 
                        help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", default=True, 
                        help="Enable Gradio sharing (default: True)")
    
    args = parser.parse_args()
    
    # Update global configuration with command line arguments
    MODEL_PATH = args.model
    DEVICE_ID = args.device
    
    print(f"Gradio version: {gr.__version__}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Device ID: {DEVICE_ID}")
    print(f"Server: {args.host}:{args.port}")
    
    demo = create_demo()
    
    try:
        demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, debug=False, show_api=False)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        stop_cleanup_thread()
        logger.info("Cleanup complete, exiting.")


if __name__ == "__main__":
    main()