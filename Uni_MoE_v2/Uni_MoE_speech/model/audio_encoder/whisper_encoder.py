import torch
import torch.nn as nn

from transformers import WhisperConfig, WhisperModel, WhisperProcessor, WhisperPreTrainedModel


class WhisperAudioTower(WhisperModel):
    def __init__(self, audio_tower, args, delay_load=False):
        config = WhisperConfig.from_pretrained(audio_tower)
        super().__init__(config)
        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')
        self.audio_split_type_dim = 4
        self.language=args.language
        self.task=args.task
        self.local_files_only=args.local_files_only

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.audio_tower_name)

    def load_model(self):
        print("load audio tower from:",self.audio_tower_name)
        self.audio_processor = WhisperProcessor.from_pretrained(self.audio_tower_name,
                                                                language = self.language,
                                                                task = self.task,
                                                                no_timestamps=True,
                                                                local_files_only = self.local_files_only)
        self.audio_tower = WhisperModel.from_pretrained(self.audio_tower_name,
                                                        local_files_only=self.local_files_only).encoder
        self.audio_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs[self.select_layer]

        return audio_features

    @torch.no_grad()
    def forward(self, audios):
        if audios.dim() == self.audio_split_type_dim:
            audio_features = []
            for k in range(audios.shape[0]):
                audio = audios[k,:,:,:]
                audio = self._mask_input_features(audio, attention_mask=None)
                # print(audio.shape)
                audio_forward_out = self.audio_tower(audio.to(device=self.device, dtype=self.dtype), output_hidden_states=True,return_dict=True)
                audio_feature = self.feature_select(audio_forward_out).to(audio.dtype)
                # print(audio_feature.shape)
                audio_features.append(audio_feature)
            audio_features = torch.cat(audio_features, dim=1)
        else:
            audio_forward_outs = self.audio_tower(audios.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            audio_features = self.feature_select(audio_forward_outs).to(audios.dtype)

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.audio_tower.config
    #     else:
    #         return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.audio_size // self.config.patch_size) ** 2
