# UniMoE Audio 推理接口

这个项目提供了多种方式来使用UniMoE Audio模型进行音频生成，包括简化的inference函数和完整的批处理框架。该接口支持三种主要任务：文本生成音乐、文本转语音（语音克隆）和视频生成音乐。

## 文件说明

- `inference.py` - 简化的inference函数，支持单个任务的快捷调用
- `inference_framework.py` - 完整的批处理框架，支持配置文件和批量任务
- `example.py` - 所有支持任务的基本使用示例
- `audio_loader.py` - 音频提示管理工具
- `test_config.json` - 框架配置文件示例
- `prompt_audios.json` - 语音克隆的音频提示数据库

## 快速开始

### 1. 使用简化的inference函数

#### 命令行使用

```bash
# 生成音乐
python inference.py --task text_to_music --input "A peaceful piano melody" --output ./music_output --model /path/to/your/model

# 语音克隆
python inference.py --task text_to_speech --input "Hello world" --ref-audio ref.wav --ref-text "Reference text" --output ./speech_output --model /path/to/your/model

# 视频生成音乐
python inference.py --task video_text_to_music --input "Upbeat electronic music" --video ./video.mp4 --output ./video_music_output --model /path/to/your/model
```

#### Python代码中使用

```python
from inference import inference

# 生成音乐
music_file = inference(
    task="text_to_music",
    input_text="A peaceful piano melody",
    output_path="./output",
    model_path="/path/to/your/model"
)

# 语音克隆
speech_file = inference(
    task="text_to_speech",
    input_text="Hello world",
    ref_audio="reference.wav",
    ref_text="Reference transcript",
    output_path="./output",
    model_path="/path/to/your/model"
)

# 视频生成音乐
video_music_file = inference(
    task="video_text_to_music",
    input_text="Upbeat electronic music",
    video_path="./video.mp4",
    output_path="./output",
    model_path="/path/to/your/model"
)
```

### 2. 使用批处理框架

#### 准备配置文件

1. 复制并修改 `config_example.json`：
```json
{
  "model_path": "/path/to/your/model",
  "device_id": 0,
  "output_base_dir": "./generated_audio",
  "log_level": "INFO",
  "log_file": "inference.log"
}
```

2. 复制并修改 `tasks_example.json`：
```json
[
  {
    "task_type": "text_to_music",
    "task_id": "music_001",
    "caption": "A peaceful piano melody",
    "output_path": "./output/music"
  },
  {
    "task_type": "text_to_speech",
    "task_id": "speech_001",
    "target_text": "Hello world",
    "reference_audio": "reference.wav",
    "reference_text": "Reference transcript",
    "output_path": "./output/speech"
  }
]
```

#### 运行批处理

```bash
python inference_framework.py --config config.json --tasks tasks.json --output-results results.json
```

## 参数说明

### inference函数参数

- `task`: 任务类型，"text_to_music" 或 "text_to_speech"
- `input_text`: 输入文本
- `ref_audio`: 参考音频文件路径（text_to_speech必需）
- `ref_text`: 参考文本（text_to_speech必需）
- `output_path`: 输出目录路径
- `model_path`: 模型路径
- `device_id`: GPU设备ID
- `reuse_model`: 是否重用模型实例（默认True）

### 命令行参数

```
--task, -t          任务类型 (text_to_music 或 text_to_speech)
--input, -i         输入文本
--ref-audio, -ra    参考音频文件路径
--ref-text, -rt     参考文本
--output, -o        输出目录路径
--model, -m         模型路径
--device, -d        GPU设备ID
--no-reuse          不重用模型实例
```

## 使用示例

### 生成多首音乐

```bash
python inference.py -t text_to_music -i "Classical symphony" -o ./music -m /path/to/model
python inference.py -t text_to_music -i "Jazz piano solo" -o ./music -m /path/to/model
python inference.py -t text_to_music -i "Electronic dance music" -o ./music -m /path/to/model
```

### 批量语音克隆

```bash
python inference.py -t text_to_speech -i "Hello, how are you?" -ra ref.wav -rt "Reference" -o ./speech -m /path/to/model
python inference.py -t text_to_speech -i "Welcome to our service" -ra ref.wav -rt "Reference" -o ./speech -m /path/to/model
```

## 注意事项

1. 确保模型路径正确
2. 对于text_to_speech任务，必须提供ref_audio和ref_text参数
3. 确保参考音频文件存在且可访问
4. 输出目录会自动创建
5. 使用`clear_model()`函数可以释放模型内存