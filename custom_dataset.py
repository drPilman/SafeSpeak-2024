from torch import distributions, nn, Tensor
from torch.utils import data
from torchaudio import functional as F
import torchaudio
import speechbrain as sb

import pandas as pd

import random
import typing as tp

class AudioDataset(data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 augmentations: tp.Optional[list[str]] = None,
                 freq_masking: int = 15,
                 time_masking: int = 15,
        ) -> None:
        # df["content"] -> path to audio
        # df["class"]      -> str transcription
        self.content = df["content"].to_list()
        self.classes = df["class"].to_list()
    
        self.augmentations = augmentations

        self.spec_mask = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(15),
            torchaudio.transforms.TimeMasking(25),
        )
    

    def len(self) -> int:
        return len(self.classes)
    

    def __getitem__(self, idx: int) -> tuple[list[str]]:
        wav, sr = torchaudio.load(self.content[idx])
        if sr != 16000:
            wav = F.resample(wav, sr, 16000)

        if random.random() > 0.8:
            wav = self.audio_aug(wav)
        
        return wav, self.classes[idx]
    

    def audio_aug(self, wav: Tensor) -> Tensor:
        if self.augmentations is not None:
            for aug_type in self.augmentations:
                if aug_type == "noise":
                    noiser = distributions.Normal(0, 4e-2)
                    return wav + noiser.sample(wav.size())
            
                elif aug_type == "speed_up":
                    pertrubator = sb.augment.time_domain.SpeedPerturb(orig_freq=16_000, speed=[90, 100, 110, 120])
                    return pertrubator(wav)

                elif aug_type == "pitch":
                    n_steps = random.choice([-3, -2, 2])
                    return F.pitch_shift(waveform=wav,
                                        sample_rate=16000,
                                        n_steps=n_steps)
            