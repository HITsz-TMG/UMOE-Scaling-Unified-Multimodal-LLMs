import os
import torch
import torch.nn as nn
import re
from .whisper_decoder import WhisperAudioDecoder

class AudioAligner(nn.Module):
    def __init__(self, audio_tower, config, **kwargs):
        super().__init__()
        self.decoder = WhisperAudioDecoder(audio_tower, args=config, **kwargs)
        projector_type = getattr(config, 'audio_projector_type', 'linear')
        if projector_type == 'linear':
            projector = nn.Linear(config.mm_audio_hidden_size, config.hidden_size)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_audio_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                projector = nn.Sequential(*modules)
        self.projector = projector
    def forward(self, encoder_output, query_tokens,**kwargs):
        decoder_output = self.decoder(encoder_output = encoder_output,query_tokens= query_tokens,**kwargs)
        projector_output = self.projector(decoder_output)
        return projector_output



def build_audio_aligner(config, **kwargs):

    audio_tower = getattr(config, 'mm_audio_tower', getattr(config, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    if is_absolute_path_exists or audio_tower.startswith("openai") or audio_tower.startswith("laion"):
        return AudioAligner(audio_tower, config=config, **kwargs)
    raise ValueError(f'Unknown audio tower: {audio_tower}')


        