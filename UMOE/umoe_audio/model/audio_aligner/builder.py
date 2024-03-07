import os
import torch
import torch.nn as nn
import re
from .Qformer import Qformer

class AudioAligner(nn.Module):
    def __init__(self, audio_tower, config, **kwargs):
        super().__init__()
        self.qformer = Qformer(audio_tower, args=config, **kwargs)
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
        self.query_tokens = nn.Parameter(torch.zeros(1, config.query_tokens_size, config.mm_audio_hidden_size))

    def forward(self, encoder_output,**kwargs):
        query_tokens = self.query_tokens.expand(encoder_output.shape[0], -1, -1)
        decoder_output = self.qformer(encoder_output = encoder_output,query_tokens= query_tokens,**kwargs)
        projector_output = self.projector(decoder_output)
        return projector_output



def build_audio_aligner(config, **kwargs):

    audio_tower = getattr(config, 'mm_audio_tower', getattr(config, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    if is_absolute_path_exists or audio_tower.startswith("openai") or audio_tower.startswith("laion"):
        return AudioAligner(audio_tower, config=config, **kwargs)
    raise ValueError(f'Unknown audio tower: {audio_tower}')


        