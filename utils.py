import sys
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch_optimizer import AdaBound
import json


def progressbar(it, prefix="", size=60, out=sys.stdout):  # Python3.6+
    count = len(it)
    start = time.time()

    def show(j):
        x = int(size * j / count)
        remaining = ((time.time() - start) / j) * (count - j)
        passing = time.time() - start
        mins_pas, sec_pass = divmod(passing, 60)
        time_pas = f"{int(mins_pas):02}:{sec_pass:05.2f}"

        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"

        print(f"{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count} time {time_pas} / {time_str}", end='\r', file=out,
              flush=True)

    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


class ChanelWiseStats(nn.Module):
    """
    The class that computes mean and standart deviation
    in input data acrocc channels
    """

    def __init__(self):
        super(ChanelWiseStats, self).__init__()

    def forward(self, x):
        x = x.view(x.data.shape[0], x.data.shape[1],
                   x.data.shape[2] * x.data.shape[3])

        mean = torch.mean(x, 2)
        std = torch.std(x, 2)

        return torch.stack((mean, std), dim=1)


class View(nn.Module):
    """
    Auxiliary class
    """

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


def pad_random(x, max_len=64600):
    x_len = x.shape[0]

    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def get_optimizer(model, config):
    if config["opt"] == 'Adam':
        optimizer = Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    elif config["opt"] == 'AdaBound':
        optimizer = AdaBound(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")
    return optimizer


def load_checkpoint(path):
    with open(path, "r") as f:
        return json.load(f)
