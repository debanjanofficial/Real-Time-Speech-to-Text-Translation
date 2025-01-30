import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class SpeechRecognitionModel(nn.Module):
    def __init__(self, config_path='configs/model_config.yaml'):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load pre-trained Wav2Vec 2.0 model
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            config['speech_recognition']['model_name']
        )
        
        # Freeze base model parameters
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config['speech_recognition'].get('num_classes', 1000))
        )
    
    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        return self.classifier(pooled_output)
