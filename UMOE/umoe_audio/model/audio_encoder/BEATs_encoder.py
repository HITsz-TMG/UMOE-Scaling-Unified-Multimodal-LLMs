import torch
import torch.nn as nn

from .BEATs import BEATs, BEATsConfig, BEATsProcessor

class BEATsAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')
        self.audio_split_type_dim = 3
        # self.language=args.language
        # self.task=args.task
        # print(args.task,args.language)
        self.local_files_only=args.local_files_only

        if not delay_load:
            self.load_model()
        else:
            checkpoint = torch.load(self.audio_tower_name)
            self.cfg_only = BEATsConfig(checkpoint['cfg'])

    def load_model(self):
        print("load audio tower from:",self.audio_tower_name)
        # print(self.task,self.language)
        # print("audio_tower:loading model",self.audio_tower_name)
        self.audio_processor = BEATsProcessor()
        checkpoint = torch.load(self.audio_tower_name)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.config = cfg
        self.audio_tower = BEATs(cfg)
        self.audio_tower.load_state_dict(checkpoint['model'])
        
        # debug
        # for k,v in self.audio_tower.named_parameters():
        #     if "conv2.weight" in k:
        #         print("aaaa",k,v)
        self.audio_tower.requires_grad_(False)
        # for name, parameter in self.audio_tower.named_parameters():
        #     print(name)
        #     print(parameter)
        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs[self.select_layer]
        # if self.select_feature == 'patch':
        #     audio_features = audio_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     audio_features = audio_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return audio_features

    @torch.no_grad()
    def forward(self, audio, padding_mask):
        if audio.dim() == self.audio_split_type_dim:
            audio_forward_out = self.audio_tower.extract_features(audio, padding_mask=padding_mask)
            audio_features = self.feature_select(audio_forward_out).to(audio.dtype)
        else:
            raise ValueError("Fbank feature wrong dimension.")
        return audio_features

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.audio_tower.dtype

    # @property
    # def device(self):
    #     return self.audio_tower.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.audio_tower.config
    #     else:
    #         return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.encoder_embed_dim

    # @property
    # def num_patches(self):
    #     return (self.config.audio_size // self.config.patch_size) ** 2
