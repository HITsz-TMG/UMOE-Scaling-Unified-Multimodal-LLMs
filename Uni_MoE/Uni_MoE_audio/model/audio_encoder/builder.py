import os
from .BEATs_encoder import BEATsAudioTower


def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    if is_absolute_path_exists or audio_tower.startswith("openai") or audio_tower.startswith("laion"):
        return BEATsAudioTower(audio_tower = audio_tower , args=audio_tower_cfg, **kwargs)

    raise ValueError(f'Unknown audio tower: {audio_tower}')
