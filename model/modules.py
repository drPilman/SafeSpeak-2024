import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=64000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=0, min_band_hz=0):

        super(SincConv, self).__init__()

        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        # In the future we will set high hz as band_hz + low + min_band_hz + min_low_hz
        # Where band_hz is (high_hz - low_hz). Therefore, it is reasonable to
        # do diff and do not set high_hz as sr/2

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))  # learnable f1 from the paper

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(
            torch.Tensor(np.diff(hz)).view(-1, 1))  # learnable f2 (f2 = f1+diff) from the paper

        # len(g) = kernel_size
        # It is symmetric, therefore we will do computations only with left part, while creating g.

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # self.window is eq. (8)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)  # eq. (5) + make sure low >= min_low_hz

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz,
                           self.sample_rate / 2)  # eq. (6) + make sure band has length >= min_band_hz
        band = (high - low)[:, 0]  # g[0] / 2

        f_times_t_low = torch.matmul(low, self.n_)  # 2 * pi * n * freq / sr
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                    self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)  # g[0] = 2 * (f2 - f1) = 2 * band, w[0] = 1
        band_pass_right = torch.flip(band_pass_left, dims=[1])  # g[n] = g[-n]

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)  # create full g[n]

        band_pass = band_pass / (2 * band[:, None])  # normalize so the max is 1

        # band_pass_left = sr * correct (4)
        # center = freq (not scaled via division) = sr * scaled_freq
        # thus, after normalization we will divide all by sr and get normalized correct(4) + normalized center

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)  # x[n] * g[n]


class ResBlock(nn.Module):
    def __init__(self, nb_filts: int, first=False):
        super().__init__()
        # for first res block in net
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x: Tensor) -> Tensor:
        # original sample save
        original_x = x

        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)

        else:
            out = x

        out = self.conv1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        # resizing original sample in case of size diff with out
        if self.downsample:
            original_x = self.conv_downsample(original_x)

        # adding original sample in the end of res block
        out += original_x
        out = self.mp(out)
        return out


class Res2Block(nn.Module):
    def __init__(self, nb_filts, first=False, nums=4):
        super(Res2Block, self).__init__()
        self.nb_filts = nb_filts
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=1,
                               padding=0,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.relu = nn.ReLU(inplace=True)
        self.nums = nums
        self.SE = SE_Block(nb_filts[1])

        convs = []
        bns = []

        for i in range(self.nums):
            convs.append(nn.Conv2d(in_channels=(nb_filts[1] // self.nums),
                                   out_channels=(nb_filts[1] // self.nums),
                                   kernel_size=3,
                                   stride=1,
                                   padding=1))
            bns.append(nn.BatchNorm2d((nb_filts[1] // self.nums)))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=1,
                               padding=0,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(nb_filts[1])

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)
        else:
            self.downsample = False

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.nb_filts[1] // self.nums, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp += spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.SE(out)

        if self.downsample:
            residual = self.conv_downsample(residual)
        out += residual
        out = self.relu(out)
        out = self.mp(out)
        return out


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class Encoder(nn.Module):
    def __init__(self, d_args, _block=ResBlock):
        super().__init__()

        # list of some args of original model. Full list: https://github.com/clovaai/aasist/blob/main/config/AASIST.conf
        self.d_args = d_args
        filts = d_args["filts"]

        self.sinc_conv = SincConv(out_channels=filts[0],
                                  kernel_size=d_args["first_conv"],
                                  )

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.res_encoder = nn.Sequential(
            _block(nb_filts=filts[1], first=True),
            _block(nb_filts=filts[2]),
            _block(nb_filts=filts[3]),
            _block(nb_filts=filts[4]),
            _block(nb_filts=filts[4]),
            _block(nb_filts=filts[4])
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.sinc_conv(x)
        x = x.unsqueeze(dim=1)

        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        e = self.res_encoder(x)
        return e