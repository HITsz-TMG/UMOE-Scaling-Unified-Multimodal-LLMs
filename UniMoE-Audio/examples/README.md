# UniMoE Audio Inference Interface

This project provides multiple ways to use the UniMoE Audio model for audio generation, including a simplified inference function and a complete batch processing framework. The interface supports three main tasks: text-to-music generation, text-to-speech with voice cloning, and video-to-music generation.

## File Overview

- `inference.py` - Simplified inference function for quick single-task calls
- `inference_framework.py` - Complete batch processing framework with configuration files
- `example.py` - Basic usage examples for all supported tasks
- `audio_loader.py` - Audio prompt management utility
- `test_config.json` - Example configuration file for the framework
- `prompt_audios.json` - Audio prompt database for voice cloning

## Quick Start

### 1. Using the Simplified Inference Function

#### Command-Line Usage

```bash
# Generate music
python inference.py --task text_to_music --input "A peaceful piano melody" --output ./music_output --model /path/to/your/model

# Voice cloning
python inference.py --task text_to_speech --input "Hello world" --ref-audio ref.wav --ref-text "Reference text" --output ./speech_output --model /path/to/your/model

# Video-to-music generation
python inference.py --task video_text_to_music --input "Upbeat electronic music" --video ./video.mp4 --output ./video_music_output --model /path/to/your/model
```

#### Programmatic Usage in Python

```python
from inference import inference

# Generate music
music_file = inference(
    task="text_to_music",
    input_text="A peaceful piano melody",
    output_path="./output",
    model_path="/path/to/your/model"
)

# Voice cloning
speech_file = inference(
    task="text_to_speech",
    input_text="Hello world",
    ref_audio="reference.wav",
    ref_text="Reference transcript",
    output_path="./output",
    model_path="/path/to/your/model"
)

# Video-to-music generation
video_music_file = inference(
    task="video_text_to_music",
    input_text="Upbeat electronic music",
    video_path="./video.mp4",
    output_path="./output",
    model_path="/path/to/your/model"
)
```

### 2. Using the Batch Processing Framework

#### Prepare Configuration Files

1. Copy and modify `test_config.json`:
```json
{
  "model_path": "/path/to/your/model",
  "device_id": 0,
  "output_base_dir": "./generated_audio",
  "log_level": "INFO",
  "log_file": "inference.log",
  "max_concurrent_tasks": 1
}
```

2. Create your task configuration file:
```json
[
  {
    "task_type": "text_to_music",
    "task_id": "music_001",
    "caption": "A peaceful piano melody",
    "output_path": "./output/music",
    "save_name": "peaceful_piano",
    "cfg_scale": 3.0,
    "temperature": 1.0
  },
  {
    "task_type": "text_to_speech",
    "task_id": "speech_001",
    "target_text": "Hello world",
    "prompt_text": "Reference transcript",
    "prompt_wav": "reference.wav",
    "output_path": "./output/speech",
    "save_name": "demo_speech",
    "cfg_scale": 3.0,
    "temperature": 1.0
  },
  {
    "task_type": "video_text_to_music",
    "task_id": "video_music_001",
    "video_path": "path/to/video.mp4",
    "caption": "Upbeat electronic music matching the video rhythm",
    "output_path": "./output/video_music",
    "save_name": "video_soundtrack",
    "fps": 8,
    "max_frames": 64,
    "cfg_scale": 3.0,
    "temperature": 1.0
  }
]
```

#### Run Batch Processing

```bash
python inference_framework.py --config config.json --tasks tasks.json --output-results results.json
```

## Parameter Descriptions

### Inference Function Parameters

- `task`: Task type, one of "text_to_music", "text_to_speech", or "video_text_to_music"
- `input_text`: Input text for generation
- `ref_audio`: Reference audio file path (required for text_to_speech)
- `ref_text`: Reference text (required for text_to_speech)
- `video_path`: Video file path (required for video_text_to_music)
- `output_path`: Output directory path
- `model_path`: Path to the model
- `device_id`: GPU device ID
- `reuse_model`: Whether to reuse the loaded model instance (default: True)

### Command-Line Parameters

```
--task, -t          Task type (text_to_music, text_to_speech, or video_text_to_music)
--input, -i         Input text for generation
--ref-audio, -ra    Reference audio file path
--ref-text, -rt     Reference text
--video, -v         Video file path (required for video_text_to_music)
--output, -o        Output directory path
--model, -m         Path to the model
--device, -d        GPU device ID
--no-reuse          Do not reuse the model instance
```

### Task Configuration Parameters

#### Common Parameters
- `task_type`: Task type ("text_to_music", "text_to_speech", or "video_text_to_music")
- `task_id`: Unique identifier for the task
- `output_path`: Output directory path
- `save_name`: Custom filename for the generated audio (optional)
- `cfg_scale`: Classifier-free guidance scale (default: 3.0)
- `temperature`: Sampling temperature (default: 1.0)

#### Text-to-Music Parameters
- `caption`: Text description of the music to generate

#### Text-to-Speech Parameters
- `target_text`: Text to be spoken
- `prompt_text`: Reference text from the prompt audio
- `prompt_wav`: Path to the reference audio file

#### Video-to-Music Parameters
- `video_path`: Path to the input video file
- `caption`: Text description of the music style
- `fps`: Frames per second for video processing (default: 8)
- `max_frames`: Maximum number of frames to process (default: 64)

## Examples

### Generate Multiple Music Tracks

```bash
python inference.py -t text_to_music -i "Classical symphony" -o ./music -m /path/to/model
python inference.py -t text_to_music -i "Jazz piano solo" -o ./music -m /path/to/model
python inference.py -t text_to_music -i "Electronic dance music" -o ./music -m /path/to/model
```

### Batch Voice Cloning

```bash
python inference.py -t text_to_speech -i "Hello, how are you?" -ra ref.wav -rt "Reference" -o ./speech -m /path/to/model
python inference.py -t text_to_speech -i "Welcome to our service" -ra ref.wav -rt "Reference" -o ./speech -m /path/to/model
```

### Video-to-Music Generation

```bash
# Generate music for a dance video
python inference.py -t video_text_to_music -i "Energetic dance music" -v dance_video.mp4 -o ./video_music -m /path/to/model

# Generate ambient music for a nature video
python inference.py -t video_text_to_music -i "Peaceful nature sounds with soft piano" -v nature_video.mp4 -o ./video_music -m /path/to/model

# Generate electronic music for a tech demo
python inference.py -t video_text_to_music -i "Futuristic electronic soundtrack" -v tech_demo.mp4 -o ./video_music -m /path/to/model
```

### Using the Audio Loader for Voice Cloning

```python
from audio_loader import AudioPromptLoader

# Load audio prompts
loader = AudioPromptLoader()

# Get available audio prompts
prompts = loader.get_audio_info()

# Use a specific prompt for voice cloning
english_male_prompt = loader.get_audio_info('english', 'male')
audio_data, sr, metadata = loader.load_audio('english', 'male')

# Generate speech with the loaded prompt
speech_file = inference(
    task="text_to_speech",
    input_text="Your custom text here",
    ref_audio=metadata['audio_path'],
    ref_text=metadata['prompt'],
    output_path="./output",
    model_path="/path/to/your/model"
)
```

## Features

- **Multi-Modal Support**: Supports text-to-music, text-to-speech, and video-to-music generation
- **Model Reuse**: Automatically caches the model instance to avoid redundant loading
- **Flexible Invocation**: Supports both command-line and Python code usage
- **Complete Parameters**: Covers all required parameters with additional configuration options
- **Error Handling**: Comprehensive error handling and exception capturing
- **Batch Processing**: Handles large-scale tasks with the batch processing framework
- **Audio Prompt Management**: Built-in audio prompt loader for voice cloning
- **Video Processing**: Automatic video frame extraction and processing for video-to-music tasks
- **Configurable Generation**: Adjustable parameters like cfg_scale, temperature, fps, and max_frames
- **Logging**: Full logging and progress tracking

## Notes

1. Ensure the model path is correct and the model is compatible with the current API
2. For text_to_speech tasks, both `prompt_wav` and `prompt_text` are required
3. For video_text_to_music tasks, ensure the video file exists and is in a supported format
4. Ensure all reference files (audio/video) exist and are accessible
5. Output directories will be created automatically
6. Use the `clear_model()` function to release model memory when done
7. Video processing requires `moviepy` library for frame extraction
8. Supported video formats depend on your system's codec availability
9. For optimal performance, use GPU acceleration when available
10. The `save_name` parameter allows custom naming of output files