import torch
from torch import nn

from .TCN import TemporalConvNet
from .capsules import PrimaryCapsules, RoutingMechanism
from .modules import Encoder, Res2Block


def get_model(config):
    if config["model"] == 'ResCapsGuard':
        model = CapsuleNet(
            num_class=config["num_class"],
            gpu_id=config["gpu_id"],
            d_args=config["d_args"],
            num_capsules=config["num_capsules"],
            num_iterations=config["num_iterations"]
        )
    elif config["model"] == 'Res2TCNGuard':
        model = Res2TCNGuard(d_args=config["d_args"])
    else:
        raise ValueError(f"Model {config['model']} not supported")
    if config["checkpoint"] is not None:
        weights = torch.load(config["checkpoint"], map_location="cpu")
        model.load_state_dict(weights)
    return model.to(config["device"])


class CapsuleNet(nn.Module):
    """
    Switch model to eval mode during validation/inference.
    """
    def __init__(self, num_class, gpu_id, d_args, num_capsules=3, num_iterations=3):
        super(CapsuleNet, self).__init__()

        self.num_class = num_class
        self.extractor = Encoder(d_args)
        self.fea_ext = PrimaryCapsules(num_capsules=num_capsules)
        self.routing_stats = RoutingMechanism(gpu_id=gpu_id,
                                              num_input_capsules=num_capsules,
                                              num_output_capsules=2,
                                              data_in=8,
                                              data_out=4,
                                              num_iterations=num_iterations)

    def forward(self, x, random=True, dropout=0.05, random_size=0.01):
        z = self.extractor(x)
        z = self.fea_ext(z)
        z = self.routing_stats(z, random, dropout, random_size=random_size)
        class_ = z.detach()
        class_ = class_.mean(dim=1)
        return z, class_


class Res2TCNGuard(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.encoder = Encoder(d_args, _block=Res2Block)
        self.tempCNN1 = TemporalConvNet(64, [72, 36, 24, 12, 6])
        self.tempCNN2 = TemporalConvNet(64, [72, 36, 24, 12, 6])
        self.relu = nn.ReLU(0.1)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.linear1 = nn.Linear(138, 4)
        self.linear2 = nn.Linear(174, 4)
        self.linear3 = nn.Linear(8, 54)
        self.linear4 = nn.Linear(54, 2)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x, random=0, dropout=0, random_size=0):
        x = self.encoder(x)
        matrix1, _ = torch.max(x, dim=2)  # T
        matrix2, _ = torch.max(x, dim=3)  # S
        x1 = self.tempCNN1(matrix2)
        x1 = torch.flatten(x1, 1, 2)
        x1 = self.linear1(x1)
        x1 = self.drop(x1)
        x1 = self.relu(x1)

        x2 = self.tempCNN2(matrix1)
        x2 = torch.flatten(x2, 1, 2)
        x2 = self.linear2(x2)
        x2 = self.drop(x2)
        x2 = self.relu(x2)

        last_layer = self.relu(self.linear3(torch.cat((x1, x2), dim=1)))
        return last_layer, self.linear4(last_layer)
