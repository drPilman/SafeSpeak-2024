import torch
from torch import distributions, Tensor
from torch.utils import data
import torchaudio
from torch_audiomentations import (
    Compose, 
    AddGaussianNoise, 
    PitchShift, 
    TimeMask,
    TimeStretch,
)
import speechbrain as sb
import soundfile as sf

import random
import typing as tp
import os

from utils import pad_random, pad
from dataset import get_data_for_dataset


class SafeSpeek(data.Dataset):
    """ASVSpoof2019 dataset rebuild"""
    def __init__(self, 
                 ids: list[int], 
                 dir_path: str, 
                 labels: list[str], 
                 pad_fn: tp.Callable = pad_random, 
                 cut: int = 64600,
                 is_train: bool = True
        ) -> None:
        """Args
        _______
        - ids (list[int]) - id's for audios
        - dir_path (str) - path to folder with audios
        - labels (list[str]) - Ground-Truth labels for audio
        - pad_fn (typing.Callable) - function for padding audios
        - cut (int) - max length of audio
        - is_train (bool) - train of eval dataset
        """
        self.ids = ids
        self.dir_path = dir_path
        self.labels = labels
        self.pad_fn = pad_fn
        self.cut = cut
        self.is_train = is_train
        if self.is_train:
            self.augmentations = Compose(
                transforms=[
                    AddGaussianNoise(
                        min_amplitude=0.001,
                        max_amplitude=0.015,
                        p=0.1
                    ),
                    PitchShift(
                        min_semitones=-2.0,
                        max_semitones=2.0,
                        p=0.1
                    ),
                    TimeMask(
                        # in %
                        min_band_part=0.1,
                        max_band_part=0.15,
                        fade=True,
                        p=0.1
                    ),
                    TimeStretch(
                        min_rate=0.8,
                        max_rate=1.2,
                        # padding
                        leave_length_unchanged=True,
                        p=0.1
                    )
                ]
            )


    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path_to_flac = f"{self.dir_path}/{self.ids[index]}.flac"
        audio, rate = sf.read(path_to_flac)
        x_pad = self.pad_fn(audio, self.cut)
        x_inp = Tensor(x_pad)

        if not self.is_train:
            return x_inp, self.ids[index], torch.tensor(self.labels[index])
        
        # TODO: Uncomment augmentations
        # x_inp = self._augment_audio(x_inp)

        return x_inp, torch.tensor(self.labels[index]), rate

    def __len__(self) -> int:
        return len(self.ids)
    
    def _augment_audio(self, audio: Tensor) -> Tensor:
        return self.augmentations(audio)
    

def get_datasets(config):
    if config["model"] == "Res2TCNGuard":
        val_pad_fn = pad
    else:
        val_pad_fn = pad_random

    train_ids, train_labels = get_data_for_dataset(config["train_label_path"])
    train_dataset = SafeSpeek(
        train_ids,
        config["train_path_flac"],
        train_labels
    )

    dev_ids, dev_labels = get_data_for_dataset(config["dev_label_path"])
    dev_dataset = SafeSpeek(
        dev_ids,
        config["dev_path_flac"],
        dev_labels,
        val_pad_fn,
        False
    )

    # eval_ids, eval_labels = get_data_for_dataset(config["eval_label_path"])

    # eval_dataset = SafeSpeek(eval_ids, config["eval_path_flac"], eval_labels, val_pad_fn, False)

    return {
        "train": train_dataset,
        "dev": dev_dataset
        # "eval": eval_dataset
    }            