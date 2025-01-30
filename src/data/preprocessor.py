import librosa
import numpy as np
from typing import Tuple, Dict

class AudioPreprocessor:
    def __init__(self, config: Dict):
        self.sample_rate = config['speech_recognition']['sampling_rate']
        self.max_length = config['speech_recognition']['max_length']
        
    def process_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Pad or truncate to max_length
        target_length = self.sample_rate * self.max_length
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
            
        return audio, self.sample_rate

class TextPreprocessor:
    def __init__(self):
        self.special_chars = {
            'á': 'ā', 'à': 'æ', 'ã': 'ä', 'â': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e',
            'í': 'ï', 'ì': 'ǐ', 'î': 'i',
            'ó': 'ö', 'ò': 'œ', 'õ': 'ǒ', 'ô': 'o',
            'ú': 'u', 'ù': 'ü', 'û': 'u',
        }
    
    def normalize_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        
        # Replace special characters
        for special, normal in self.special_chars.items():
            text = text.replace(special, normal)
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
