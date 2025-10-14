# -*- coding: utf-8 -*-
"""
UniMoE-Audio Utils Package

This package contains the core utilities for UniMoE-Audio model:
- UniMoE_Audio_utils: DAC utilities, DeepSpeed MoE inference utilities, and matrix compression utilities
- UniMoE_Audio_core: MoE core components including UniMoEAudioSparseMoeBlock and related classes
- UniMoE_Audio_model: Main model classes including UniAudioRVQQwen2_5VLMoEForConditionalGeneration
- UniMoE_Audio_mod: High-level interface and convenience functions

Usage:
    # Quick start with high-level interface
    from utils.UniMoE_Audio_mod import UniMoEAudio
    
    # Load model
    model = UniMoEAudio("path/to/model")
    
    # Generate music
    audio = model.text_to_music("A peaceful piano melody")
    
    # Generate speech
    audio = model.text_to_speech("Hello world", prompt_text="prompt text", prompt_wav="prompt audio path")
    
    # Generate music from video
    audio = model.video_text_to_music("video.mp4", "A soundtrack for this video")

    # Low-level access to components
    from utils import UniMoEAudioSparseMoeBlock, UniAudioRVQQwen2_5VLMoEForConditionalGeneration
"""

import warnings
from typing import Optional, List, Dict, Any

# Version information
__version__ = "1.0.0"
__author__ = "UniMoE-Audio HITsz-TMG"
__description__ = "UniMoE-Audio: A Unified Speech and Music Generation with Dynamic-Capacity Mixture of Experts"

# Import DAC and audio utilities
try:
    from .UniMoE_Audio_utils import (
        # DAC utilities
        Dac,
        build_delay_indices,
        apply_audio_delay,
        build_revert_indices,
        revert_audio_delay,
        _prepare_audio_prompt,
        
        # DeepSpeed utilities
        DecoderOutput,
        _generate_output,
        
        # Matrix compression utilities
        compress_matrix,
        decompress_matrix,
        
        # MoE gating utilities
        top2gating,
        gate_forward,
        
        # Vision utilities
        Conv3D,
        Qwen2_5_VisionPatchEmbed,
        Qwen2_5_VisionTransformerPretrainedModel,
    )
except ImportError as e:
    warnings.warn(f"Failed to import from UniMoE_Audio_utils: {e}")
    # Define dummy classes to prevent import errors
    class Dac: pass
    class DecoderOutput: pass
    class Conv3D: pass
    class Qwen2_5_VisionPatchEmbed: pass
    class Qwen2_5_VisionTransformerPretrainedModel: pass
    def build_delay_indices(*args, **kwargs): pass
    def apply_audio_delay(*args, **kwargs): pass
    def build_revert_indices(*args, **kwargs): pass
    def revert_audio_delay(*args, **kwargs): pass
    def _prepare_audio_prompt(*args, **kwargs): pass
    def _generate_output(*args, **kwargs): pass
    def compress_matrix(*args, **kwargs): pass
    def decompress_matrix(*args, **kwargs): pass
    def top2gating(*args, **kwargs): pass
    def gate_forward(*args, **kwargs): pass

# Import core MoE components
try:
    from .UniMoE_Audio_core import (
        # MoE blocks
        UniMoEAudioSparseMoeBlock,
        UniMoEAudioMoE,
        AudioMOELayer,
        AudioExperts,
        
        # Expert selection and routing
        audio_sparse_expert_mixer,
        audio_dynamic_expert_selection,
        calculate_audio_global_routing_weight,
        
        # Loss functions
        audio_load_balancing_loss_func,
        
        # MLP variants
        AudioSharedExpertMLP,
        AudioDynamicExpertMLP,
        AudioNullExpertMLP,
        
        # Routing functions
        AudioMoERoutingFunction,
    )
except ImportError as e:
    warnings.warn(f"Failed to import from UniMoE_Audio_core: {e}")
    # Define dummy classes
    class UniMoEAudioSparseMoeBlock: pass
    class UniMoEAudioMoE: pass
    class AudioMOELayer: pass
    class AudioExperts: pass
    class AudioSharedExpertMLP: pass
    class AudioDynamicExpertMLP: pass
    class AudioNullExpertMLP: pass
    class AudioMoERoutingFunction: pass
    def audio_load_balancing_loss_func(*args, **kwargs): pass
    def audio_sparse_expert_mixer(*args, **kwargs): pass
    def audio_dynamic_expert_selection(*args, **kwargs): pass
    def calculate_audio_global_routing_weight(*args, **kwargs): pass

# Import model classes
try:
    from .UniMoE_Audio_model import (
        # Configuration classes
        UniAudioRVQQwen2_5VLMoEConfig,
        Qwen2_5_VLMoETextConfig,
        
        # Model classes
        UniAudioRVQQwen2_5VLMoEForConditionalGeneration,
        Qwen2_5_VLMoEDecoderLayer,
        Qwen2_5_VLMoETextModel,
        Qwen2_5_VLMoEPreTrainedModel,
        
        # Output classes
        MoEQwen2_5VLCausalLMOutputWithPast,
        BaseModelOutputWithPast,
    )
except ImportError as e:
    warnings.warn(f"Failed to import from UniMoE_Audio_model: {e}")
    # Define dummy classes
    class UniAudioRVQQwen2_5VLMoEConfig: pass
    class Qwen2_5_VLMoETextConfig: pass
    class UniAudioRVQQwen2_5VLMoEForConditionalGeneration: pass
    class Qwen2_5_VLMoEDecoderLayer: pass
    class Qwen2_5_VLMoETextModel: pass
    class Qwen2_5_VLMoEPreTrainedModel: pass
    class MoEQwen2_5VLCausalLMOutputWithPast: pass
    class BaseModelOutputWithPast: pass

# Import high-level interface
try:
    from .UniMoE_Audio_mod import (
        # Main interface class
        UniMoEAudio,
    )
except ImportError as e:
    warnings.warn(f"Failed to import from UniMoE_Audio_mod: {e}")
    # Define dummy classes
    class UniMoEAudio: pass

# Define comprehensive __all__ list
__all__ = [
    # Version and metadata
    "__version__",
    "__author__", 
    "__description__",
    
    # DAC and audio utilities
    "Dac",
    "build_delay_indices",
    "apply_audio_delay", 
    "build_revert_indices",
    "revert_audio_delay",
    "_prepare_audio_prompt",
    
    # DeepSpeed utilities
    "DecoderOutput",
    "_generate_output",
    
    # Matrix compression utilities
    "compress_matrix",
    "decompress_matrix",
    
    # MoE gating utilities
    "top2gating",
    "gate_forward",
    
    # Vision utilities
    "Conv3D",
    "Qwen2_5_VisionPatchEmbed",
    "Qwen2_5_VisionTransformerPretrainedModel",
    
    # Core MoE components
    "UniMoEAudioSparseMoeBlock",
    "UniMoEAudioMoE",
    "AudioMOELayer", 
    "AudioExperts",
    "audio_load_balancing_loss_func",
    "audio_sparse_expert_mixer",
    "audio_dynamic_expert_selection",
    "calculate_audio_global_routing_weight",
    
    # MLP variants
    "AudioSharedExpertMLP",
    "AudioDynamicExpertMLP", 
    "AudioNullExpertMLP",
    
    # Routing functions
    "AudioMoERoutingFunction",
    
    # Configuration classes
    "UniAudioRVQQwen2_5VLMoEConfig",
    "Qwen2_5_VLMoETextConfig",
    
    # Model classes
    "UniAudioRVQQwen2_5VLMoEForConditionalGeneration",
    "Qwen2_5_VLMoEDecoderLayer",
    "Qwen2_5_VLMoETextModel", 
    "Qwen2_5_VLMoEPreTrainedModel",
    
    # Output classes
    "MoEQwen2_5VLCausalLMOutputWithPast",
    "BaseModelOutputWithPast",
    
    # High-level interface
    "UniMoEAudio",
]

# Convenience aliases for backward compatibility
UnimoeAudio = UniMoEAudio  # Alternative naming

# Add aliases to __all__
__all__.extend(["UnimoeAudio"])

def get_version() -> str:
    """Get the current version of UniMoE Audio."""
    return __version__

def list_available_models() -> List[str]:
    """List available model types."""
    return [
        "UniAudioRVQQwen2_5VLMoEForConditionalGeneration",
        "Qwen2_5_VLMoETextModel",
    ]

def get_model_info() -> Dict[str, Any]:
    """Get information about the UniMoE Audio package."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "available_models": list_available_models(),
        "components": {
            "utils": "DAC utilities, matrix compression, DeepSpeed utilities, vision components",
            "core": "MoE components, expert routing, load balancing",
            "model": "Main model classes and configurations", 
            "mod": "High-level interface and convenience functions"
        }
    }

# Add utility functions to __all__
__all__.extend(["get_version", "list_available_models", "get_model_info"])

# Package initialization message
def _show_welcome_message():
    """Show welcome message when package is imported."""
    try:
        print(f"🎵 UniMoE Audio v{__version__} loaded successfully!")
        print("📖 Quick start: from utils.UniMoE_Audio_mod import UniMoEAudio")
        print("🔗 Documentation: Use help(utils) for more information")
    except:
        pass  # Silently fail if print is not available

# Show welcome message on import (can be disabled by setting environment variable)
import os
if not os.environ.get("UNIMOE_AUDIO_QUIET", False):
    _show_welcome_message()