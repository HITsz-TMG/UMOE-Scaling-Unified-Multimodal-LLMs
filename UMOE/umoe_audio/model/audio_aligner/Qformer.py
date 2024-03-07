import torch
import torch.nn as nn

from transformers import (
    Blip2QFormerModel,
)
from .blip2_config import Blip2Config

class Qformer(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')
        self.local_files_only = args.local_files_only

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = Blip2Config()

        


    def load_model(self):
        config = Blip2Config()
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.qformer.requires_grad_(True)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.last_hidden_state
        # if self.select_feature == 'patch':
        #     audio_features = audio_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     audio_features = audio_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return audio_features

    # @torch.no_grad()
    def forward(self, encoder_output, query_tokens):
        attention_mask = torch.ones(encoder_output.size()[:-1], dtype = torch.long, device = self.device)
        audio_decoder_outs = self.qformer(query_embeds = query_tokens.to(device=self.device, dtype=self.dtype),
                                                encoder_hidden_states=encoder_output,
                                                encoder_attention_mask = attention_mask,
                                                output_hidden_states=True,
                                                return_dict=True,
                                                )
        audio_features = self.feature_select(audio_decoder_outs).to(encoder_output.dtype)
        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.qformer.dtype

    @property
    def device(self):
        return self.qformer.device

    @property
    def config(self):
        if self.is_loaded:
            return self.qformer.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.audio_size // self.config.patch_size) ** 2
