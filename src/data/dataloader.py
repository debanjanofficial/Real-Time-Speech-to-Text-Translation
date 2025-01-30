import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List
from .preprocessor import AudioPreprocessor, TextPreprocessor

class SpeechTranslationDataset(Dataset):
    def __init__(self, metadata_path: str, config: Dict):
        self.metadata = pd.read_csv(metadata_path)
        self.audio_processor = AudioPreprocessor(config)
        self.text_processor = TextPreprocessor()
        
    def __len__(self) -> int:
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        
        # Process audio
        audio, sr = self.audio_processor.process_audio(row['audio_path'])
        
        # Process text
        source_text = self.text_processor.normalize_text(row['source_text'])
        target_text = self.text_processor.normalize_text(row['target_text'])
        
        return {
            'audio': torch.FloatTensor(audio),
            'source_text': source_text,
            'target_text': target_text,
            'sample_rate': sr
        }

def create_dataloaders(
    train_metadata: str,
    val_metadata: str,
    config: Dict
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = SpeechTranslationDataset(train_metadata, config)
    val_dataset = SpeechTranslationDataset(val_metadata, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['speech_recognition']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['speech_recognition']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
