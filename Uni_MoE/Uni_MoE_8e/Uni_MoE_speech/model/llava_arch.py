#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .audio_encoder.builder import build_audio_tower
from .audio_aligner.builder import build_audio_aligner

from Uni_MoE_speech.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, AUDIO_TOKEN_INDEX, VIDEO_TOKEN_INDEX



class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)###  need change delay_load
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_audio_tower"):
            self.audio_tower = build_audio_tower(config, delay_load=True)
            self.mm_audio_aligner = build_audio_aligner(config)
        if hasattr(config, "mm_query_tokens"):
            audio_tower = getattr(self, 'audio_tower', None)
            if audio_tower is not None:
                self.query_tokens = nn.Parameter(torch.zeros(1, config.query_tokens_size, audio_tower.hidden_size))

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            print("mm projector:")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        mm_audio_select_layer = model_args.mm_audio_select_layer
        mm_audio_select_feature = model_args.mm_audio_select_feature
        pretrain_audio_aligner = model_args.pretrain_audio_aligner
        query_tokens_size = model_args.query_tokens_size

        self.config.mm_audio_tower = audio_tower

        if self.get_audio_tower() is None:
            audio_tower = build_audio_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.audio_tower = [audio_tower]
            else:
                self.audio_tower = audio_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                audio_tower = self.audio_tower[0]
            else:
                audio_tower = self.audio_tower
            audio_tower.load_model()

        self.config.use_mm_proj = True
        self.config.audio_projector_type = getattr(model_args, 'audio_projector_type', 'linear')
        self.config.mm_audio_hidden_size = audio_tower.hidden_size
        self.config.mm_audio_select_layer = mm_audio_select_layer
        self.config.mm_audio_select_feature = mm_audio_select_feature
        self.config.query_tokens_size = query_tokens_size
        self.config.local_files_only = model_args.local_files_only
        self.config.language=model_args.language
        self.config.task=model_args.task
        self.config.mm_query_tokens = True

        if getattr(self, 'query_tokens', None) is None:
            self.query_tokens = nn.Parameter(torch.zeros(1, query_tokens_size, audio_tower.hidden_size))

        if getattr(self, 'mm_audio_aligner', None) is None:
            self.mm_audio_aligner = build_audio_aligner(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_audio_aligner.parameters():
                p.requires_grad = True

        if pretrain_audio_aligner is not None:
            print("audio aligner: ")
            audio_aligner_weights = torch.load(pretrain_audio_aligner, map_location='cpu')
            # for k,v in audio_aligner_weights.items():
            #     print(k)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_audio_aligner.load_state_dict(get_w(audio_aligner_weights, 'mm_audio_aligner'))
            self.query_tokens = nn.Parameter([v for k, v in audio_aligner_weights.items()][0])



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    # audios: (chunk_num,batch_size,30,8000) out: (batch_size,(query_tokens_size[50]),llm_hidden_size)
    def encode_audios(self,audios):
        encoder_output = self.get_model().get_audio_tower()(audios)
        query_tokens = self.get_model().query_tokens.expand(encoder_output.shape[0], -1, -1)
        aligner_output = self.get_model().mm_audio_aligner(encoder_output = encoder_output,query_tokens = query_tokens)
        # print(aligner_output.shape)
        return aligner_output

    def prepare_inputs_labels_for_audio_and_vision(
        self, input_ids, attention_mask, past_key_values, labels, images, image_mask, videos, video_mask, input_features, features_mask
    ):

        vision_tower = self.get_vision_tower()
        audio_tower = self.get_audio_tower()
        if (vision_tower is None and audio_tower is None) or (images is None and  input_features is None and videos is None) or (input_ids.shape[1] == 1 ):
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None

        if images is not None:
            if type(images) is list or images.ndim == 5:
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                image_features = [x.flatten(0, 1) for x in image_features]
            else:
                image_features = self.encode_images(images)
                new_image_features = []
                for blidx,bl in enumerate(image_mask):
                    if bl:
                        new_image_features.append(image_features[blidx])
                image_features = new_image_features

        if videos is not None:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_features = self.encode_images(concat_videos)
            split_sizes = [video.shape[0] for video in videos]
            video_features = torch.split(video_features, split_sizes, dim=0)
            video_features = [x.mean(dim=0,dtype=videos.dtype) for x in video_features]

            new_video_features = []
            for blidx,bl in enumerate(video_mask):
                if bl:
                    new_video_features.append(video_features[blidx])
            video_features = new_video_features
           
        
        if input_features is not None:
            audio_embeds_batch = []
            for batch_idx, cur_input_features in enumerate(input_features):
                
                cur_features_mask = features_mask[batch_idx]
                tag = 1
                audio_embeds = []
                tmp_embeds = []
                aucheck = []
                for pidx,cfm_id in enumerate(cur_features_mask):
                    
                    if cfm_id < 1: break
                    pfeatures = cur_input_features[pidx,:,:]
                    audio_embed = self.encode_audios(pfeatures.unsqueeze(dim = 0).unsqueeze(dim = 0))
                    tmp_embeds.append(audio_embed.squeeze(dim = 0))
                    aucheck.append(pfeatures)
                    if pidx == cur_features_mask.shape[0]-1 or cfm_id != cur_features_mask[pidx+1]:
                        audio_embeds.append(torch.cat(tmp_embeds, dim=0))
                        tmp_embeds=[]
                audio_embeds_batch.append(audio_embeds)

        all_image_indice=[]
        all_audio_indice=[]
        all_video_indice=[]
        all_text_indice=[]
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        cur_video_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]
            video_token_indices = torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0]
            
            cur_audio_idx = 0

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            now_place_idx = 0
            batch_image_indice=[]
            batch_audio_indice=[]
            batch_video_indice=[]
            
            while image_token_indices.numel() > 0 or audio_token_indices.numel() > 0 or video_token_indices.numel() > 0:
                if image_token_indices.numel() == 0:
                    # print('imgend')
                    image_token_start = cur_input_ids.shape[0]+1
                else:
                    image_token_start = image_token_indices[0]
                if audio_token_indices.numel() == 0:
                    # print('audend')
                    audio_token_start = cur_input_ids.shape[0]+1
                else:
                    audio_token_start = audio_token_indices[0]
                if video_token_indices.numel() == 0:
                    # print('vidend')
                    video_token_start = cur_input_ids.shape[0]+1
                else:
                    video_token_start = video_token_indices[0]
                # if input_ids:[...audio...image...] cut for audio
                if audio_token_start<image_token_start and audio_token_start < video_token_start:
                    
                    cur_audio_embeds = audio_embeds_batch[batch_idx]
                    cur_audio_in_conv = cur_audio_embeds[cur_audio_idx]

                    indice_start = now_place_idx+audio_token_start
                    indice_end = indice_start+cur_audio_in_conv.shape[0]
                    batch_audio_indice.append((int(indice_start),int(indice_end)))
                    now_place_idx=indice_end

                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:audio_token_start]))
                    cur_new_input_embeds.append(cur_audio_in_conv)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:audio_token_start]) # changed???? wrong????
                        cur_new_labels.append(torch.full((cur_audio_in_conv.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[audio_token_start+1:]

                    cur_audio_idx += 1
                    cur_input_ids = cur_input_ids[audio_token_start+1:]
                    audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                    video_token_indices = torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0]
                    

                elif image_token_start < audio_token_start and image_token_start < video_token_start:

                    cur_image_features = image_features[cur_image_idx]

                    indice_start = now_place_idx+image_token_start
                    indice_end = indice_start+cur_image_features.shape[0]
                    batch_image_indice.append((int(indice_start), int(indice_end)))
                    now_place_idx=indice_end

                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)

                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                    
                    cur_image_idx += 1
                    # dd
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                    audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                    video_token_indices = torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0]
                

                elif video_token_start < audio_token_start and video_token_start < image_token_start:

                    cur_video_features = video_features[cur_video_idx]

                    indice_start = now_place_idx+video_token_start
                    indice_end = indice_start+cur_video_features.shape[0]
                    batch_video_indice.append((int(indice_start),int(indice_end)))
                    now_place_idx=indice_end

                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:video_token_start]))
                    cur_new_input_embeds.append(cur_video_features)

                    if labels is not None:
                        cur_new_labels.append(cur_labels[:video_token_start])
                        cur_new_labels.append(torch.full((cur_video_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[video_token_start+1:]
                    
                    cur_video_idx += 1
                    # dd
                    cur_input_ids = cur_input_ids[video_token_start+1:]
                    audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                    video_token_indices = torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                # dd
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))

                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            new_input_embeds.append(cur_new_input_embeds)

            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

            all_image_indice.append(batch_image_indice)
            all_video_indice.append(batch_video_indice)
            all_audio_indice.append(batch_audio_indice)
            
            # the other part of input_ids except image , video and audio is text
            # cur_input_ids_range = range(0, cur_new_input_embeds.shape[0])

            # indices_to_remove = set()
            # for start, end in batch_image_indice:
            #     indices_to_remove.update(range(start, end))
            # for start, end in batch_audio_indice:
            #     indices_to_remove.update(range(start, end))
            # for start, end in batch_video_indice:
            #     indices_to_remove.update(range(start, end))

            # batch_text_indice = []
            # start = None
            # for i in cur_input_ids_range:
            #     if i not in indices_to_remove:
            #         if start is None:
            #             start = i
            #     elif start is not None:
            #         batch_text_indice.append((start, i))
            #         start = None
            # if start is not None:
            #     batch_text_indice.append((start, cur_new_input_embeds.shape[0]))
            
            # # print("batch_ind:",indices_to_remove,batch_text_indice,batch_audio_indice,batch_video_indice,batch_image_indice)

            # all_text_indice.append(batch_text_indice)
        all_text_indice = []
        max_embeds_len = max(x.shape[0] for x in new_input_embeds)
        for indice_batch,_ in enumerate(all_image_indice):
            n_img_i = all_image_indice[indice_batch]
            n_aud_i = all_audio_indice[indice_batch]
            n_vid_i = all_video_indice[indice_batch]
            # the other part of input_ids except image , video and audio is text
            cur_input_ids_range = range(0, max_embeds_len+1)

            indices_to_remove = set()
            for start, end in n_img_i:
                indices_to_remove.update(range(start, end))
            for start, end in n_aud_i:
                indices_to_remove.update(range(start, end))
            for start, end in n_vid_i:
                indices_to_remove.update(range(start, end))

            batch_text_indice = []
            start = None
            for i in cur_input_ids_range:
                if i not in indices_to_remove:
                    if start is None:
                        start = i
                elif start is not None:
                    batch_text_indice.append((start, i))
                    start = None
            if start is not None:
                batch_text_indice.append((start, max_embeds_len+1))
            all_text_indice.append(batch_text_indice)
        # print("batch_ind:",batch_text_indice,batch_audio_indice,batch_video_indice,batch_image_indice)
        return_indice = {
            "image":all_image_indice,
            "audio":all_audio_indice,
            "video":all_video_indice,
            "text":all_text_indice
        }
        # if different shape, pad to the same shape and get attention mask
        # attention mask padd right so [11111100000] after we insert images and audios it becomes 
        # (audio,image increase length)*[1]+[111111100000]+(pad length)*[0]
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, return_indice

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
