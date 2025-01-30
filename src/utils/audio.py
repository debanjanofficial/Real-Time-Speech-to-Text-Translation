import numpy as np
import librosa
from typing import Tuple

def load_audio(
    file_path: str,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """Load and resample audio file"""
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def compute_melspectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 80
) -> np.ndarray:
    """Compute mel spectrogram from audio"""
    melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=160,
        win_length=400
    )
    melspec = librosa.power_to_db(melspec)
    return melspec
