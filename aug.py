import torch
import torchaudio
from torchaudio.io import AudioEffector
import random
import yaml
import pyroomacoustics as pra
import numpy as np
import torchaudio.functional as F
from audiomentations import (
    Compose,
    AddGaussianNoise,
    PitchShift,
    TimeMask,
    TimeStretch,
)
from pathlib import Path
import torch

def align_waveform(wav1, wav2):
    assert wav2.size(1) >= wav1.size(1)
    diff = wav2.size(1) - wav1.size(1)
    min_mse = float("inf")
    best_i = -1

    for i in range(diff):
        segment = wav2[:, i : i + wav1.size(1)]
        mse = torch.mean((wav1 - segment) ** 2).item()
        if mse < min_mse:
            min_mse = mse
            best_i = i

    return best_i, wav2[:, best_i : best_i + wav1.size(1)]
def get_audio_paths(path: str):
    start_dir = Path(path).resolve()
    return list(start_dir.rglob('*.wav'))

class Degrader:
    def __init__(self):

        with open("configs/aug.yaml", "r") as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        self.use_rir = self.yaml["use"]["use_rir"]
        self.use_noise = self.yaml["use"]["use_noise"]
        self.use_codec = self.yaml["use"]["use_codec"]
        self.use_spectr = self.yaml["use"]["use_spectr"]
        self.use_phone = self.yaml["use"]["use_phone"]

        self.snr_min = self.yaml["snr_range"]["min"]
        self.snr_max = self.yaml["snr_range"]["max"]

        if self.use_rir:
            self.rirs = []
            self.prepare_rir(self.yaml["room_info"]["count"])

        if self.use_noise:
            self.noise_paths = get_audio_paths(self.yaml["paths"]["noise_path"])

        if self.use_codec:
            self.codecs = list(self.yaml["codecs"].keys())

    def _add_codec(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ):
        codec_name = random.choice(self.codecs)
        config = self.yaml["codecs"][codec_name]
        encoder = AudioEffector(**config)

        codec_wav = encoder.apply(waveform.T, sample_rate).T

        if waveform.size(1) != codec_wav.size(1):
            best_idx, codec_wav = align_waveform(waveform, codec_wav)

        return codec_wav.float()

    def prepare_rir(self, n_rirs):
        for _ in range(n_rirs):
            cfg_room = self.yaml["room_info"]
            x_min, x_max = cfg_room["x_min"], cfg_room["x_max"]
            z_min, z_max = cfg_room["z_min"], cfg_room["z_max"]
            x = random.uniform(x_min, x_max)
            y = random.uniform(x_min, x_max)
            z = random.uniform(z_min, z_max)
            corners = np.array([[0, 0], [0, y], [x, y], [x, 0]]).T
            room = pra.Room.from_corners(corners, max_order=10, absorption=0.2)
            room.extrude(z)
            room.add_source(cfg_room["src_pos"])
            room.add_microphone(cfg_room["micr_pos"])

            room.compute_rir()
            rir = torch.tensor(np.array(room.rir[0]))
            rir = rir / rir.norm(p=2)
            self.rirs.append(rir)

    def _add_rir(self, waveform, sample_rate):
        if len(self.rirs) == 0:
            raise RuntimeError
        rir = random.choice(self.rirs)
        augmented = torchaudio.functional.fftconvolve(waveform, rir)
        if waveform.size(1) != augmented.size(1):
            augmented = augmented[:, : waveform.size(1)]
        return augmented.float()

    def _add_noise(self, waveform, sample_rate):
        if len(self.noise_paths) == 0:
            raise RuntimeError

        snr_max, snr_min = self.snr_max, self.snr_min
        snr = random.uniform(snr_min, snr_max)
        noise_path = random.choice(self.noise_paths)
        noise, noise_sr = torchaudio.load(noise_path)
        noise /= noise.norm(p=2)

        if noise.size(0) > 1:
            noise = noise[0].unsqueeze(0)
        noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)

        if not noise.size(1) < waveform.size(1):
            start_idx = random.randint(0, noise.size(1) - waveform.size(1))
            end_idx = start_idx + waveform.size(1)
            noise = noise[:, start_idx:end_idx]
        else:
            noise = noise.repeat(1, waveform.size(1) // noise.size(1) + 1)[
                :, : waveform.size(1)
            ]

        augmented = torchaudio.functional.add_noise(
            waveform=waveform, noise=noise, snr=torch.tensor([snr])
        )
        return augmented

    def _apply_sp_deg(self, waveform, sample_rate):
        waveform = waveform.T
        num_aug = random.randint(0, self.yaml["spectrogramm"]["num_aug"])

        for _ in range(num_aug):
            effect = random.choice(self.yaml["spectrogramm"]["effects"])
            effector = AudioEffector(effect=effect, pad_end=True)
            waveform = effector.apply(waveform, sample_rate)

        return waveform.T

    def _add_phone(self, waveform, sample_rate):
        waveform = waveform.T
        effect = ",".join(
            [
                "lowpass=frequency=4000:poles=1",
                "compand=attacks=0.02:decays=0.05:points=-60/-60|-30/-10|-20/-8|-5/-8|-2/-8:gain=-8:volume=-7:delay=0.05",
            ]
        )
        effector = AudioEffector(effect=effect, format="g722")
        return effector.apply(waveform, sample_rate).T

    def __call__(self, waveform:torch.Tensor, sample_rate=16000):
        waveform = waveform.unsqueeze(0)
        codec = False

        if random.random() < self.yaml["probs"]["rir_prob"] and self.use_rir:
            waveform = self._add_rir(waveform, sample_rate)

        if random.random() < self.yaml["probs"]["noise_prob"] and self.use_noise:
            waveform = self._add_noise(waveform, sample_rate)

        if random.random() < self.yaml["probs"]["codec_prob"] and self.use_codec:
            codec = True
            waveform = self._add_codec(waveform, sample_rate)

        if random.random() < self.yaml["probs"]["specrt_prob"] and self.use_spectr:
            waveform = self._apply_sp_deg(waveform, sample_rate)

        if (
            random.random() < self.yaml["probs"]["phone_prob"]
            and self.use_phone
            and not (codec)
        ):
            waveform = self._add_phone(waveform, sample_rate)
        waveform = waveform.squeeze()
        return waveform

class Degrader2:
    def __init__(self):
        self.augmentations = Compose(
            transforms=[
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.2),
                TimeMask(
                    min_band_part=0.1,
                    max_band_part=0.15,
                    fade=True,
                    p=0.1,
                ),
                TimeStretch(
                    min_rate=0.8,
                    max_rate=1.2,
                    leave_length_unchanged=True,
                    p=0.1,
                ),
            ]
        )
        with open("configs/aug.yaml", "r") as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)
        self.use_rir = self.yaml["use"]["use_rir"]
        if self.use_rir:
            self.rirs = []
            self.prepare_rir(self.yaml["room_info"]["count"])

    def prepare_rir(self, n_rirs):
        for _ in range(n_rirs):
            cfg_room = self.yaml["room_info"]
            x_min, x_max = cfg_room["x_min"], cfg_room["x_max"]
            z_min, z_max = cfg_room["z_min"], cfg_room["z_max"]
            x = random.uniform(x_min, x_max)
            y = random.uniform(x_min, x_max)
            z = random.uniform(z_min, z_max)
            corners = np.array([[0, 0], [0, y], [x, y], [x, 0]]).T
            room = pra.Room.from_corners(corners, max_order=10, absorption=0.2)
            room.extrude(z)
            room.add_source(cfg_room["src_pos"])
            room.add_microphone(cfg_room["micr_pos"])

            room.compute_rir()
            rir = torch.tensor(np.array(room.rir[0]))
            rir = rir / rir.norm(p=2)
            self.rirs.append(rir)

    def _add_rir(self, waveform, sample_rate):
        if len(self.rirs) == 0:
            raise RuntimeError
        rir = random.choice(self.rirs)
        augmented = torchaudio.functional.fftconvolve(waveform, rir)
        if waveform.size(1) != augmented.size(1):
            augmented = augmented[:, : waveform.size(1)]
        return augmented.float()

    def __call__(self, waveform: np.ndarray) -> torch.Tensor:
        waveform = torch.Tensor(self.augmentations(waveform, 16000))
        # if random.random() < 0.5:
        #     audio_segment = AudioSegment(
        #         waveform.tobytes(),
        #         frame_rate=16000,
        #         sample_width=waveform.dtype.itemsize,
        #         channels=1
        #     )
        #     audio_segment.export(encoded, format="ogg")
        if random.random() < self.yaml["probs"]["rir_prob"] and self.use_rir:
            waveform = self._add_rir(waveform.unsqueeze(0), 16000).squeeze()
        return waveform
