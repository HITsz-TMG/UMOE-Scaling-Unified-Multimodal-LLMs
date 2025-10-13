# -*- coding: utf-8 -*-
"""
UniMoE-Audio mod    
"""

import os
import sys
import math
import time
import tempfile
import shutil
from typing import List, Optional, Union
from pathlib import Path

import torch
import torchaudio
import torchvision
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, AutoProcessor

from qwen_vl_utils import smart_resize

# Import from UniMoE_Audio_utils modules
from .UniMoE_Audio_utils import (
    Dac,
    _prepare_audio_prompt,
    DecoderOutput,
    _generate_output,
)
from .UniMoE_Audio_model import (
    UniAudioRVQQwen2_5VLMoEForConditionalGeneration,
    UniAudioRVQQwen2_5VLMoEConfig,
)

class UniMoEAudio: 
    def __init__(self, model_path: str, device_id: int = 0):
        """
        Initialize UniMoE Audio model
        """
        # Configuration parameters
        self.TORCH_DTYPE = torch.bfloat16
        self.MAX_TOKENS = 1000
        self.MIN_TOKENS = 100
        
        # Video processing constants
        self.IMAGE_FACTOR = 28
        self.VIDEO_TOTAL_PIXELS = 512 * 28 * 28
        self.VIDEO_MIN_PIXELS = 16 * 28 * 28
        self.VIDEO_MAX_PIXELS = 64 * 28 * 28
        self.FRAME_FACTOR = 2
        
        # Templates and constants
        self.SYSTEM_MESSAGE = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"""
        self.INPUT_FORMAT = """<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""
        self.AUDIO_START = "<|AUDIO_START|>"
        self.DEFAULT_VIDEO_PROMPT = "{}"
        
        # Create a temporary directory
        self.TEMP_DIR = tempfile.mkdtemp()
        
        # Initialize model components
        self._initialize_model(model_path, device_id)
    
    def _initialize_model(self, model_path: str, device_id: int):
        """Initialize model, DAC, and tokenizer"""
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading UniMoE Audio model...")
        try:
            self.model = UniAudioRVQQwen2_5VLMoEForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.TORCH_DTYPE,
                attn_implementation="sdpa"
            ).to(self.device)
            print("Using SDPA attention implementation")
        except Exception as e:
            print(f"SDPA failed, falling back to eager attention: {e}")
            self.model = UniAudioRVQQwen2_5VLMoEForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.TORCH_DTYPE,
                attn_implementation="eager"
            ).to(self.device)
            print("Using eager attention implementation")
        self.model.eval()
        
        # Load DAC
        print("Loading DAC...")
        self.dac = Dac()
        self._move_dac_to_device(self.dac, self.device)
        
        # Load tokenizer and processor
        print("Loading Tokenizer and Processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            use_fast=False
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer = self.tokenizer
        
        # Special tokens
        special_tokens = [
            "<|AUDIO_PLACEHOLDER|>", "<|AUDIO_START|>", "<|AUDIO_END|>",
            "<|SPEECH_START|>", "<|SPEECH_END|>", 
            "<|VOICE_PROMPT_START|>", "<|VOICE_PROMPT_END|>",
            "<|SPEECH_PROMPT_START|>", "<|SPEECH_PROMPT_END|>",
            "<|MUSIC_START|>", "<|MUSIC_END|>"
        ]
        
        # Validate special tokens
        assert all(len(self.tokenizer([t]).input_ids[0]) == 1 for t in special_tokens)
        
        print("Model initialization complete!")
    
    def _move_dac_to_device(self, dac_instance, target_device):
        """Move DAC components to the target device"""
        if hasattr(dac_instance, 'model') and dac_instance.model is not None:
            dac_instance.model.to(target_device)
        if hasattr(dac_instance, 'resampler'):
            for key in dac_instance.resampler:
                dac_instance.resampler[key].to(target_device)
    
    def __del__(self):
        """Clean up the temporary directory"""
        try:
            if hasattr(self, 'TEMP_DIR') and self.TEMP_DIR and os.path.exists(self.TEMP_DIR):
                shutil.rmtree(self.TEMP_DIR, ignore_errors=True)
        except (AttributeError, TypeError):
            pass
    
    def _preprocess_codec(self, codec, codec_delay_pattern, codec_channels, 
                         codec_bos_value, codec_eos_value, codec_pad_value):
        """Preprocess codec tokens"""
        codec_token = torch.tensor(codec, dtype=torch.long)
        codec_token_len = codec_token.shape[0]
        max_delay_pattern = max(codec_delay_pattern)
        codec_input_ids = torch.zeros((codec_token_len + max_delay_pattern + 1, codec_channels), dtype=torch.long)
        
        for c in range(codec_channels):
            start = codec_delay_pattern[c] + 1
            codec_input_ids[:start, c] = codec_bos_value
            codec_input_ids[start : start + codec_token_len, c] = codec_token[:, c]
            codec_input_ids[start + codec_token_len :, c] = codec_pad_value
            if start + codec_token_len < codec_input_ids.shape[0]:
                codec_input_ids[start + codec_token_len, c] = codec_eos_value
        
        return codec_input_ids
    
    def _frame_process(self, images, **kwargs):
        """Process video frames"""
        images = [torchvision.transforms.functional.pil_to_tensor(img) for img in images]
        video = torch.stack(images, dim=0)
        
        nframes, _, height, width = video.shape
        min_pixels = kwargs.get("min_pixels", self.VIDEO_MIN_PIXELS)
        total_pixels = kwargs.get("total_pixels", self.VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(self.VIDEO_MAX_PIXELS, total_pixels / nframes * self.FRAME_FACTOR), 
                        int(min_pixels * 1.05))
        max_pixels_supposed = kwargs.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            print(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        
        if "resized_height" in kwargs and "resized_width" in kwargs:
            resized_height, resized_width = smart_resize(
                kwargs["resized_height"],
                kwargs["resized_width"],
                factor=self.IMAGE_FACTOR,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        
        return video
    
    def _extract_images_from_video(self, video_path: str, fps: int, max_frames: Optional[int] = None):
        """Extract image frames from a video"""
        video = VideoFileClip(video_path)
        duration = video.duration
        
        images = []
        for i, t in enumerate(range(0, math.ceil(duration * fps))):
            time_in_video = t / fps
            frame = video.get_frame(time_in_video)
            img = Image.fromarray(frame)
            images.append(img)
            
            if max_frames is not None and i >= max_frames - 1:
                break
        
        video.close()
        return images
    
    def _generate_core(self, source_input, codec_input_ids, save_name: str, rebuild_codec=None, output_dir: str = "./",
                      cfg_scale: float = 0.0, temperature: float = 1.0, top_p: float = 1.0, 
                      cfg_filter_top_k: int = 45, eos_prob_mul_factor: float = 0.6, 
                      do_sample: bool = True, debug_guidance_step: int = -1, use_cache: bool = True):
        """Core generation function with configurable audio generation hyperparameters
        
        Args:
            source_input: Input data for generation
            codec_input_ids: Codec input IDs
            save_name: Name for saving the generated audio
            rebuild_codec: Optional codec for rebuilding
            output_dir: Directory to save output files
            cfg_scale: Classifier-free guidance scale (default: 0.0)
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 1.0)
            cfg_filter_top_k: Top-k filtering for CFG (default: 45)
            eos_prob_mul_factor: EOS probability multiplication factor (default: 0.6)
            do_sample: Whether to use sampling (default: True)
            debug_guidance_step: Debug guidance step (default: -1)
            use_cache: Whether to use cache (default: True)
        """
        batch_size = source_input.input_ids.shape[0] // 2
        
        prefill, prefill_steps = _prepare_audio_prompt(self.model, audio_prompts=[None] * batch_size)
        labels_prefill = None
        dec_output = DecoderOutput(prefill, prefill_steps, self.model.device, labels_prefill=labels_prefill)
        
        with torch.no_grad():
            generated_codes, lengths_Bx = self.model.generate(
                input_ids=source_input.input_ids.to(self.model.device),
                codec_input_ids=codec_input_ids.to(self.model.device) 
                            if codec_input_ids is not None else None,
                
                pixel_values=source_input.pixel_values.to(self.model.device) 
                            if hasattr(source_input, 'pixel_values') and source_input.pixel_values is not None else None,
                pixel_values_videos=source_input.pixel_values_videos.to(self.model.device) 
                            if hasattr(source_input, 'pixel_values_videos') and source_input.pixel_values_videos is not None else None,
                image_grid_thw=source_input.image_grid_thw.to(self.model.device) 
                            if hasattr(source_input, 'image_grid_thw') and source_input.image_grid_thw is not None else None,
                video_grid_thw=source_input.video_grid_thw.to(self.model.device) 
                            if hasattr(source_input, 'video_grid_thw') and source_input.video_grid_thw is not None else None,
                second_per_grid_ts=source_input.second_per_grid_ts.to(self.model.device) 
                            if hasattr(source_input, 'second_per_grid_ts') and source_input.second_per_grid_ts is not None else None,
                
                attention_mask=source_input.attention_mask.to(self.model.device),
                dec_output=dec_output,
                max_tokens=self.MAX_TOKENS,
                min_tokens=self.MIN_TOKENS,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                eos_prob_mul_factor=eos_prob_mul_factor,
                do_sample=do_sample,
                debug_guidance_step=debug_guidance_step,
                use_cache=use_cache,
            )
        
        audios = _generate_output(self.model, generated_codes, lengths_Bx)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save generated audios
        output_paths = []
        for i in range(len(audios)):
            output_path = os.path.join(output_dir, f"generated_{save_name}_{i}.wav")
            self.dac.decode(audios[i].transpose(0, 1).unsqueeze(0), save_path=output_path, min_duration=1)
            output_paths.append(output_path)
            print(f"Generated audio saved to: {output_path}")
            
            if rebuild_codec is not None:
                codec = torch.tensor(rebuild_codec).transpose(0, 1)
                rebuild_path = os.path.join(output_dir, f"rebuild_{save_name}_{i}.wav")
                self.dac.decode(codec.unsqueeze(0), save_path=rebuild_path)
                print(f"Rebuild audio saved to: {rebuild_path}")
        
        return output_paths
    
    def text_to_music(
            self, caption: Union[str, List[str]], 
            save_name: str = "music", 
            rebuild_codec=None, 
            output_dir: str = "./",
            cfg_scale: float = 10.0, 
            temperature: float = 1.0, 
            top_p: float = 1.0, 
            cfg_filter_top_k: int = 45, 
            eos_prob_mul_factor: float = 0.6, 
            do_sample: bool = True, 
            debug_guidance_step: int = -1, 
            use_cache: bool = True
        ) -> List[str]:
        """
        Text-to-music generation
        v 
        Args:
            caption: Music description text, can be a string or a list of strings
            save_name: File prefix for saving
            rebuild_codec: Optional codec for reconstruction
            output_dir: Directory to save the generated audio files
            cfg_scale: Classifier-free guidance scale (default: 0.0)
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 1.0)
            cfg_filter_top_k: Top-k filtering for CFG (default: 45)
            eos_prob_mul_factor: EOS probability multiplication factor (default: 0.6)
            do_sample: Whether to use sampling (default: True)
            debug_guidance_step: Debug guidance step (default: -1)
            use_cache: Whether to use cache (default: True)
        """
         # Input validation and processing
        if isinstance(caption, str):
            if not caption.strip():
                print("Please enter a music description.")
                return []
            caption = [caption]
        else:
            # Filter out empty captions
            caption = [c for c in caption if c.strip()]
            if not caption:
                print("Please enter valid music descriptions.")
                return []
        
        print(f"Input: Generating music for {len(caption)} description(s)")
        for i, desc in enumerate(caption):
            print(f"  [{i+1}] {desc}")
        
        print("Preparing text prompts...")
        neg_text = self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format("<|MUSIC_START|>Low quality.<|MUSIC_END|>") + self.AUDIO_START
        text_input = []
        
        for i in range(len(caption)):
            text_input.append(neg_text)
            text_input.append(self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format("<|MUSIC_START|>" + caption[i] + "<|MUSIC_END|>") + self.AUDIO_START)
        
        print("Tokenizing input text...")
        source_input = self.tokenizer(text_input, add_special_tokens=False, return_tensors="pt", padding=True)
        
        print("Starting music generation...")
        start_time = time.time()
        
        output_paths = self._generate_core(source_input, None, save_name, rebuild_codec, output_dir,
                                         cfg_scale, temperature, top_p, cfg_filter_top_k, 
                                         eos_prob_mul_factor, do_sample, debug_guidance_step, use_cache)
        
        generation_time = time.time() - start_time
        print(f"Music generation completed! Time: {generation_time:.2f}s")
        print(f"Output: Generated {len(output_paths)} audio file(s):")
        for i, path in enumerate(output_paths):
            print(f"  [{i+1}] {path}")
        
        return output_paths
    
    def text_to_speech(
            self, 
            caption: Union[str, List[str]], 
            prompt_text: str, 
            prompt_wav: str, 
            save_name: str = "speech", 
            prompt_codec=None, 
            rebuild_codec=None, 
            output_dir: str = "./",
            cfg_scale: float = 1.0, 
            temperature: float = 1.0, 
            top_p: float = 1.0, 
            cfg_filter_top_k: int = 45, 
            eos_prob_mul_factor: float = 1.0, 
            do_sample: bool = True, 
            debug_guidance_step: int = -1, 
            use_cache: bool = True
        ) -> List[str]:
        """
        Text-to-speech generation (voice cloning)
        
        Args:
            caption: Target speech text, can be a string or a list of strings
            prompt_text: Reference audio text content
            prompt_wav: Path to the reference audio file
            save_name: File prefix for saving
            prompt_codec: Optional pre-encoded codec of reference audio
            rebuild_codec: Optional codec for reconstruction
            output_dir: Directory to save the generated audio files
            cfg_scale: Classifier-free guidance scale (default: 0.0)
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 1.0)
            cfg_filter_top_k: Top-k filtering for CFG (default: 45)
            eos_prob_mul_factor: EOS probability multiplication factor (default: 0.6)
            do_sample: Whether to use sampling (default: True)
            debug_guidance_step: Debug guidance step (default: -1)
            use_cache: Whether to use cache (default: True)
        """
        # Input validation and processing
        if isinstance(caption, str):
            if not caption.strip():
                print("Please enter the target text.")
                return []
            caption = [caption]
        else:
            # Filter out empty captions
            caption = [c for c in caption if c.strip()]
            if not caption:
                print("Please enter valid target texts.")
                return []
        
        if prompt_wav is None:
            print("Please provide a reference audio file.")
            return []
        
        if not prompt_text.strip():
            print("Please enter the reference audio transcript.")
            return []
        
        print(f"Input: Generating speech for {len(caption)} text(s)")
        for i, text in enumerate(caption):
            print(f"  [{i+1}] {text}")
        print(f"Reference audio: {prompt_wav}")
        print(f"Reference text: {prompt_text}")
        
        # Encode reference audio
        print("Encoding reference audio...")
        if prompt_codec is None:
            assert prompt_wav is not None
            prompt_codec = self.dac.encode(prompt_wav)
        
        print("Preprocessing codec...")
        prompt_codec_input_ids = self._preprocess_codec(
            codec=prompt_codec,
            codec_delay_pattern=self.model.config.codec_delay_pattern,
            codec_channels=self.model.num_channels,
            codec_bos_value=self.model.config.codec_bos_value,
            codec_eos_value=self.model.config.codec_eos_value,
            codec_pad_value=self.model.config.codec_pad_value
        )
        
        # Construct prompt text
        print("Preparing text prompts...")
        prompt_caption = "<|SPEECH_PROMPT_START|>" + prompt_text + "<|SPEECH_PROMPT_END|>"
        prompt_caption += "<|VOICE_PROMPT_START|>" + "<|AUDIO_PLACEHOLDER|>" * prompt_codec_input_ids.shape[0] + "<|VOICE_PROMPT_END|>"
        
        prompt_caption_fn = lambda x: prompt_caption + "<|SPEECH_START|>" + x + "<|SPEECH_END|>"
        
        neg_text = self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format(prompt_caption_fn("")) + self.AUDIO_START
        text_input = []
        
        for i in range(len(caption)):
            text_input.append(neg_text)
            text_input.append(self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format(prompt_caption_fn(caption[i])) + self.AUDIO_START)
        
        print("Tokenizing input text...")
        source_input = self.tokenizer(text_input, add_special_tokens=False, return_tensors="pt", padding=True)
        
        prompt_codec_input_ids = prompt_codec_input_ids.unsqueeze(0).expand(len(text_input), -1, -1).reshape(-1, prompt_codec_input_ids.shape[1])
        
        print("Starting speech generation...")
        start_time = time.time()
        
        output_paths = self._generate_core(source_input, prompt_codec_input_ids, save_name, rebuild_codec, output_dir,
                                         cfg_scale, temperature, top_p, cfg_filter_top_k, 
                                         eos_prob_mul_factor, do_sample, debug_guidance_step, use_cache)
        
        generation_time = time.time() - start_time
        print(f"Speech generation completed! Time: {generation_time:.2f}s")
        print(f"Output: Generated {len(output_paths)} audio file(s):")
        for i, path in enumerate(output_paths):
            print(f"  [{i+1}] {path}")
        
        return output_paths
    
    def video_text_to_music(
                self, 
                video: Union[str, List[str]], 
                caption: Union[str, List[str]], 
                save_name: str = "video_music", 
                rebuild_codec=None, 
                fps: int = 1, 
                sampling_fps: int = 1, 
                total_pixels: int = 3 * 28 * 28, 
                max_frames: Optional[int] = 1,
                output_dir: str = "./",
                cfg_scale: float = 10.0, 
                temperature: float = 1.0, 
                top_p: float = 1.0,        
                cfg_filter_top_k: int = 45, 
                eos_prob_mul_factor: float = 0.6, 
                do_sample: bool = True, 
                debug_guidance_step: int = -1, 
                use_cache: bool = True
        ) -> List[str]:
        """
        Video+Text to music generation
        
        Args:
            video: Video file path(s), can be a string or a list of strings
            caption: Music description text(s), can be a string or a list of strings
            save_name: File prefix for saving
            rebuild_codec: Optional codec for reconstruction
            fps: Output FPS
            sampling_fps: Sampling FPS
            total_pixels: Total pixels
            max_frames: Maximum number of frames
            output_dir: Directory to save the generated audio files
            cfg_scale: Classifier-free guidance scale (default: 0.0)
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 1.0)
            cfg_filter_top_k: Top-k filtering for CFG (default: 45)
            eos_prob_mul_factor: EOS probability multiplication factor (default: 0.6)
            do_sample: Whether to use sampling (default: True)
            debug_guidance_step: Debug guidance step (default: -1)
            use_cache: Whether to use cache (default: True)
        """
        print("Starting video+text to music generation task...")
        
        # Input validation and conversion
        if isinstance(video, str):
            video = [video]
        if isinstance(caption, str):
            caption = [caption]
        
        assert len(video) == len(caption), "The number of videos and captions must match"
        
        print(f"Input: {len(video)} video(s)")
        for i, v in enumerate(video):
            print(f"  Video {i+1}: {v}")
        
        print(f"Input: {len(caption)} description(s)")
        for i, c in enumerate(caption):
            print(f"  Description {i+1}: {c}")
        
        print(f"Generation parameters: fps={fps}, sampling_fps={sampling_fps}, max_frames={max_frames}")
        print(f"Model parameters: cfg_scale={cfg_scale}, temperature={temperature}, top_p={top_p}")
        
        print("Preparing video and text inputs...")
        
        neg_text = self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format(
            self.DEFAULT_VIDEO_PROMPT.format("<|MUSIC_START|>Low quality.<|MUSIC_END|>")
        ) + self.AUDIO_START
        
        text_input = []
        video_inputs = []
        fps_inputs = []
        
        for i in range(len(caption)):
            print(f"Processing video-text pair {i+1}/{len(caption)}...")
            
            text_input.append(neg_text)
            text_input.append(self.SYSTEM_MESSAGE + self.INPUT_FORMAT.format(
                self.DEFAULT_VIDEO_PROMPT.format("<|MUSIC_START|>" + caption[i] + "<|MUSIC_END|>")
            ) + self.AUDIO_START)
            
            print(f"Text prompt prepared: {caption[i]}")
            
            # Process video input
            print(f"Extracting video frames: {video[i]}")
            extracted_frames = self._extract_images_from_video(video[i], sampling_fps, max_frames)
            print(f"Extracted {len(extracted_frames)} frames")
            
            print("Processing video frames...")
            video_input = self._frame_process(
                extracted_frames,
                fps=fps,
                total_pixels=total_pixels,
                min_pixels=1 * 28 * 28
            )
            
            video_inputs.append(video_input)
            video_inputs.append(video_input)
            
            fps_inputs.append(fps)
            fps_inputs.append(fps)
            
            num_frames, _, resized_height, resized_width = video_input.shape
            print(f"Video processing completed - shape: {video_input.shape}")
            print(f"Video token count: {int(num_frames / 2 * resized_height / 28 * resized_width / 28)}")
            print(f"FPS setting: {fps}")
        
        print("Preparing input data with processor...")
        source_input = self.processor(
            text=text_input, 
            images=None, 
            videos=video_inputs, 
            fps=fps_inputs, 
            padding=True, 
            return_tensors="pt", 
            do_resize=False
        )
        print("Input data preparation completed")
        
        print("Starting music generation...")
        start_time = time.time()
        
        result = self._generate_core(source_input, None, save_name, rebuild_codec, output_dir,
                                   cfg_scale, temperature, top_p, cfg_filter_top_k, 
                                   eos_prob_mul_factor, do_sample, debug_guidance_step, use_cache)
        
        end_time = time.time()
        print(f"Generation completed! Time: {end_time - start_time:.2f}s")
        
        if isinstance(result, list):
            print(f"Successfully generated {len(result)} audio file(s):")
            for i, path in enumerate(result):
                print(f"  File {i+1}: {path}")
        else:
            print(f"Generated audio file: {result}")
        
        return result


# Convenience function
def create_unimoe_audio(model_path: str, device_id: int = 0) -> UniMoEAudio:
    return UniMoEAudio(model_path, device_id)
