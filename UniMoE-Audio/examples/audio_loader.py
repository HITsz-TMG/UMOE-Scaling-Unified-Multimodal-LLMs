"""
Prompt demo Audio Loader for UniMoE-Audio Project
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
import numpy as np
import soundfile as sf


class AudioPromptLoader:
    def __init__(self, config_path: str = None, base_dir: str = None):
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(project_root, "examples", "prompt_audios.json")
        
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
        
        self.config_path = config_path
        self.base_dir = Path(base_dir)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Configuration file format error: {e}")
    
    def get_audio_info(self, language: str = None, gender: str = None) -> Dict:
        lang_map = {'en': 'english', 'zh': 'chinese'}
        if language in lang_map:
            language = lang_map[language]
        
        audio_prompts = self.config.get('audio_prompts', {})
        
        if language and language in audio_prompts:
            if gender and gender in audio_prompts[language]:
                return audio_prompts[language][gender]
            else:
                return audio_prompts[language]
        
        return audio_prompts
    
    def load_audio(self, language: str, gender: str, 
                   sr: int = 22050, normalize: bool = True) -> Tuple[np.ndarray, int, Dict]:
        audio_info = self.get_audio_info(language, gender)
        
        if not audio_info or 'audio_path' not in audio_info:
            raise ValueError(f"Audio information not found: language={language}, gender={gender}")
        
        audio_path = self.base_dir / audio_info['audio_path']
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        audio_data, sample_rate = librosa.load(str(audio_path), sr=sr)
        
        if normalize:
            audio_data = librosa.util.normalize(audio_data)
        
        return audio_data, sample_rate, audio_info
    
    def get_random_audio(self, language: str = None, 
                        sr: int = 22050, normalize: bool = True) -> Tuple[np.ndarray, int, Dict]:
        audio_prompts = self.config.get('audio_prompts', {})
        
        if language:
            lang_map = {'en': 'english', 'zh': 'chinese'}
            if language in lang_map:
                language = lang_map[language]
            
            if language not in audio_prompts:
                raise ValueError(f"Unsupported language: {language}")
            
            available_langs = [language]
        else:
            available_langs = list(audio_prompts.keys())
        
        selected_lang = random.choice(available_langs)
        available_genders = list(audio_prompts[selected_lang].keys())
        selected_gender = random.choice(available_genders)
        
        return self.load_audio(selected_lang, selected_gender, sr, normalize)
    
    def get_all_audio_paths(self) -> List[Dict]:
        all_audios = []
        audio_prompts = self.config.get('audio_prompts', {})
        
        for lang, genders in audio_prompts.items():
            for gender, info in genders.items():
                audio_info = info.copy()
                audio_info['full_path'] = str(self.base_dir / info['audio_path'])
                audio_info['language_key'] = lang
                audio_info['gender_key'] = gender
                all_audios.append(audio_info)
        
        return all_audios
    
    def save_audio(self, audio_data: np.ndarray, output_path: str, 
                   sr: int = 22050, format: str = 'wav') -> None:
        sf.write(output_path, audio_data, sr, format=format.upper())
    
    def get_metadata(self) -> Dict:
        return self.config.get('metadata', {})
    
    def list_available_options(self) -> Dict:
        audio_prompts = self.config.get('audio_prompts', {})
        options = {
            'languages': list(audio_prompts.keys()),
            'genders': [],
            'total_samples': 0
        }
        
        for lang, genders in audio_prompts.items():
            for gender in genders.keys():
                if gender not in options['genders']:
                    options['genders'].append(gender)
                options['total_samples'] += 1
        
        return options
