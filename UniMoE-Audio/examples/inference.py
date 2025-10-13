#!/usr/bin/env python3
"""
Simplified UniMoE Audio Inference Interface

Provides a simple function interface for audio generation tasks.
Supports both programmatic calls and command-line usage.
"""

import os
import argparse
import sys
from typing import Optional
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from utils.UniMoE_Audio_mod import UniMoEAudio


# Global model instance for reuse
_model_instance = None
_current_model_path = None


def inference(
    task: str,
    input_text: str,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    output_path: str = "./output",
    model_path: str = "path/to/your/model",
    device_id: int = 0,
    reuse_model: bool = True,
    video_path: Optional[str] = None
) -> Optional[str]:
    """
    Simplified inference function for UniMoE Audio generation.
    
    Args:
        task: Task type, either "text_to_music", "text_to_speech", or "video_text_to_music"
        input_text: Input text for generation
        ref_audio: Reference audio file path (required for text_to_speech)
        ref_text: Reference text (required for text_to_speech)
        output_path: Output directory path
        model_path: Path to the model
        device_id: GPU device ID
        reuse_model: Whether to reuse the loaded model instance
        video_path: Video file path (required for video_text_to_music)
    
    Returns:
        Path to the generated audio file, or None if failed
    """
    global _model_instance, _current_model_path
    
    try:
        # Initialize or reuse model
        if not reuse_model or _model_instance is None or _current_model_path != model_path:
            print(f"Loading model from {model_path}...")
            _model_instance = UniMoEAudio(model_path=model_path, device_id=device_id)
            _current_model_path = model_path
            print("Model loaded successfully!")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Execute task
        if task == "text_to_music":
            print(f"Generating music: {input_text}")
            result = _model_instance.text_to_music(
                caption=input_text,
                save_name="inference_music",
                output_dir=output_path,
                cfg_scale=10.0,
                temperature=1.0
            )
            
        elif task == "text_to_speech":
            if not ref_audio or not ref_text:
                raise ValueError("ref_audio and ref_text are required for text_to_speech task")
            
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
            
            print(f"Generating speech: {input_text}")
            result = _model_instance.text_to_speech(
                caption=input_text,
                prompt_text=ref_text,
                prompt_wav=ref_audio,
                save_name="inference_speech",
                output_dir=output_path,
                cfg_scale=1.0,
                temperature=1.0
            )
            
        elif task == "video_text_to_music":
            if not video_path:
                raise ValueError("video_path is required for video_text_to_music task")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            print(f"Generating music from video and text: {input_text}")
            result = _model_instance.video_text_to_music(
                video=video_path,
                caption=input_text,
                save_name="inference_video_music",
                output_dir=output_path,
                fps=1,
                max_frames=1,
                cfg_scale=10.0,
                temperature=1.0
            )
            
        else:
            raise ValueError(f"Unknown task type: {task}. Must be 'text_to_music', 'text_to_speech', or 'video_text_to_music'")
        
        if result:
            print(f"Generation completed: {result}")
            return result
        else:
            print("Generation failed")
            return None
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


def clear_model():
    """
    Clear the cached model instance to free memory.
    """
    global _model_instance, _current_model_path
    _model_instance = None
    _current_model_path = None
    print("Model instance cleared")


def main():
    """
    Command-line interface for the inference function.
    """
    parser = argparse.ArgumentParser(
        description="UniMoE Audio Inference - Simple Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate music
  python inference.py --task text_to_music --input "A peaceful piano melody" --output ./music_output
  
  # Generate speech with voice cloning
  python inference.py --task text_to_speech --input "Hello world" --ref-audio ref.wav --ref-text "Reference text" --output ./speech_output
  
  # Specify custom model path
  python inference.py --task text_to_music --input "Jazz music" --model /path/to/model --output ./output
"""
    )
    
    parser.add_argument(
        "--task", "-t",
        required=True,
        choices=["text_to_music", "text_to_speech", "video_text_to_music"],
        help="Task type: text_to_music, text_to_speech, or video_text_to_music"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input text for generation"
    )
    
    parser.add_argument(
        "--ref-audio", "-ra",
        help="Reference audio file path (required for text_to_speech)"
    )
    
    parser.add_argument(
        "--ref-text", "-rt",
        help="Reference text (required for text_to_speech)"
    )
    
    parser.add_argument(
        "--video", "-v",
        help="Video file path (required for video_text_to_music)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory path (default: ./output)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="path/to/your/model",
        help="Path to the model (default: path/to/your/model)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )
    
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Don't reuse model instance (reload for each call)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.task == "text_to_speech":
        if not args.ref_audio or not args.ref_text:
            print("Error: --ref-audio and --ref-text are required for text_to_speech task")
            return 1
    elif args.task == "video_text_to_music":
        if not args.video:
            print("Error: --video is required for video_text_to_music task")
            return 1
    
    # Call inference function
    result = inference(
        task=args.task,
        input_text=args.input,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_path=args.output,
        model_path=args.model,
        device_id=args.device,
        reuse_model=not args.no_reuse,
        video_path=args.video
    )
    
    if result:
        print(f"\nSuccess! Generated file: {result}")
        return 0
    else:
        print("\nFailed to generate audio")
        return 1


if __name__ == "__main__":
    exit(main())