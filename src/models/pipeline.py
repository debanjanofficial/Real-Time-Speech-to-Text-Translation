import torch
from typing import Dict, List

class TranslationPipeline:
    def __init__(self, speech_model, translation_model, config_path='configs/model_config.yaml'):
        self.speech_model = speech_model
        self.translation_model = translation_model
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def process(self, audio_input: torch.Tensor) -> Dict[str, str]:
        # Speech recognition
        with torch.no_grad():
            speech_output = self.speech_model(audio_input)
            
        # Translation
        translated_text = self.translation_model(speech_output)
        
        return {
            'source_text': speech_output,
            'translated_text': translated_text
        }
    
    def process_batch(self, audio_batch: List[torch.Tensor]) -> List[Dict[str, str]]:
        results = []
        for audio in audio_batch:
            result = self.process(audio)
            results.append(result)
        return results
