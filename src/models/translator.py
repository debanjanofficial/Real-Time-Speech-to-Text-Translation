import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer

class TranslationModel(nn.Module):
    def __init__(self, config_path='configs/model_config.yaml'):
        super().__init__()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.model_name = config['translation']['model_name']
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def forward(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Generate translation
        translated = self.model.generate(**inputs)
        
        # Decode translation
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        return translated_text
