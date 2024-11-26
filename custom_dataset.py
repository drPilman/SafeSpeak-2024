import torch
from torch import Tensor
from torch.utils import data
import soundfile as sf

import typing as tp

from utils import pad_random, pad
from aug import Degrader, Degrader2
from pathlib import Path
import numpy as np
import pandas as pd


augs = {"Degrader": Degrader, "Degrader2": Degrader2}


class SafeSpeek(data.Dataset):
    """ASVSpoof2019 dataset rebuild"""

    def __init__(
        self,
        paths: list[str],
        dir_path: Path,
        labels: list[str],
        pad_fn: tp.Callable = pad_random,
        cut: int = 64600,
        is_train: bool = True,
        aug: str | None = None,
        is_test: bool = False,
    ) -> None:
        self.paths = paths
        self.dir_path = dir_path
        self.labels = labels
        self.pad_fn = pad_fn
        self.cut = cut
        self.is_train = is_train
        if self.is_train and aug:
            self.aug = augs[aug]()
        self.is_test = is_test

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path = self.paths[index]
        label = torch.tensor(self.labels[index])
        if self.is_test:
            audio = np.random.rand(1, 16000)
        else:
            audio, _ = sf.read(str(self.dir_path / path), samplerate=16000, channels=1)
        x_pad = self.pad_fn(audio, self.cut)
        x_inp = Tensor(x_pad)

        if self.is_train:
            x_inp = self.aug(x_inp)

        return x_inp, label

    def __len__(self) -> int:
        return len(self.paths)


def get_data_for_dataset(csv_path):
    df = pd.read_csv(csv_path, sep=",")
    return df["path"].to_list(), df["label"].to_list()


def get_datasets(config):
    dir = Path(config["data_dir"])

    train_paths, train_labels = get_data_for_dataset(config["datasets"]["train"])
    train_dataset = SafeSpeek(
        train_paths,
        dir,
        train_labels,
        is_train=True,
        aug=config["aug"],
        is_test=config["test"],
    )

    validate_path, validate_labels = get_data_for_dataset(config["datasets"]["validate"])
    dev_dataset = SafeSpeek(
        validate_path, dir, validate_labels, is_train=False, is_test=config["test"]
    )

    # eval_ids, eval_labels = get_data_for_dataset(config["eval_label_path"])

    # eval_dataset = SafeSpeek(eval_ids, config["eval_path_flac"], eval_labels, val_pad_fn, False)

    return {
        "train": train_dataset,
        "validate": dev_dataset,
        # "eval": eval_dataset
    }
