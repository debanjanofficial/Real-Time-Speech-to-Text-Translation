import asyncio
from typing import Optional, Callable
import numpy as np

class AudioStreamHandler:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 4096):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer = []
        self.processing_callback: Optional[Callable] = None
    
    async def process_chunk(self, chunk: bytes) -> None:
        """Process incoming audio chunks"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(chunk, dtype=np.float32)
        
        # Add to buffer
        self.buffer.append(audio_data)
        
        # Process if buffer is full
        if len(self.buffer) * self.chunk_size >= self.sample_rate:
            if self.processing_callback:
                # Combine chunks and process
                full_audio = np.concatenate(self.buffer)
                await self.processing_callback(full_audio)
            # Clear buffer
            self.buffer = []
    
    def set_processing_callback(self, callback: Callable) -> None:
        """Set callback for processing complete audio segments"""
        self.processing_callback = callback
