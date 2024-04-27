import torch
import torch.nn as nn

from transformers import WhisperConfig, WhisperModel, WhisperProcessor


class WhisperAudioDecoder(nn.Module):
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
            self.cfg_only = WhisperConfig.from_pretrained(self.audio_tower_name)


    def load_model(self):
        self.audio_decoder = WhisperModel.from_pretrained(self.audio_tower_name,
                                                        local_files_only=self.local_files_only).decoder
        self.audio_decoder.requires_grad_(True)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.last_hidden_state
        return audio_features

    # @torch.no_grad()
    def forward(self, encoder_output, query_tokens):
        audio_decoder_outs = self.audio_decoder(inputs_embeds = query_tokens.to(device=self.device, dtype=self.dtype),
                                                encoder_hidden_states=encoder_output,
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
        return self.audio_decoder.dtype

    @property
    def device(self):
        return self.audio_decoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_decoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.audio_size // self.config.patch_size) ** 2
