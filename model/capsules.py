import torch
from torch import nn
from torch.autograd import Variable

from utils import ChanelWiseStats, View
import torch.nn.functional as F


class PrimaryCapsules(nn.Module):
    """
    This class create capsules and makes
    forward propagation through them
    """

    def __init__(self, num_capsules=10):
        super(PrimaryCapsules, self).__init__()

        self.num_capsules = num_capsules

        self.capsules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                ChanelWiseStats(),
                nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(8),
                nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(1),
                View(-1, 8)
            )
            for _ in range(num_capsules)
        ])

    def forward(self, x):
        results = [capsule(x) for capsule in self.capsules]
        result = torch.stack(results, dim=-1)
        return result


class RoutingMechanism(nn.Module):
    def __init__(self,
                 gpu_id,
                 num_input_capsules,
                 num_output_capsules,
                 data_in,
                 data_out,
                 num_iterations=2):
        super(RoutingMechanism, self).__init__()

        self.gpu_id = gpu_id
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(torch.randn(
            num_output_capsules, num_input_capsules,
            data_out, data_in
        ))

    def squash(self, x, dim):
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / (torch.sqrt(squared_norm))

    def forward(self, x, random, dropout, random_size=0.01):
        # x[batch, data, in_caps]

        x = x.transpose(2, 1)
        # x[batch, in_caps, data]

        if random:
            noise = Variable(random_size * torch.randn(*self.route_weights.size()))
            if self.gpu_id >= 0:
                noise = noise.cuda(self.gpu_id)
            route_weights = self.route_weights + noise  # w_ji + rand(size(w_ji))
        else:
            route_weights = self.route_weights

        priors = route_weights[:, None, :, :, :] @ x[None, :, :, :, None]

        # route_weights[out_caps, 1,     in_caps, data_out, data_in]
        # x            [1,        batch, in_caps, data_in, 1]
        # priors       [out_caps, batch, in_caps, data_out, 1]

        priors = self.squash(priors.transpose(1, 0), dim=3)  # sqush(w_ij u_i)
        # priors[batch, out_caps, in_caps, data_out, 1]

        if dropout > 0.0:
            drop = Variable(torch.FloatTensor(*priors.size())).bernoulli(1.0 - dropout)
            if self.gpu_id >= 0:
                drop = drop.cuda(self.gpu_id)
            priors = priors * drop

        logits = Variable(torch.zeros(*priors.size()))  # initialization b_ij =
        # logits[batch,out_caps,in_caps, data_out,1]

        if self.gpu_id >= 0:
            logits = logits.cuda(self.gpu_id)

        num_iterations = self.num_iterations

        for i in range(num_iterations):  # for r iterations do
            probs = F.softmax(logits, dim=2)  # a_j = softmax(b_j)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True), dim=3)  # v_j=squash(a_ji * u_ji)

            logits = priors * outputs  # b_ij = v_j*u_ji

        # outputs[b, out_caps, 1, data_out, 1]
        outputs = outputs.squeeze()

        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2, 1).contiguous()
        else:
            outputs = outputs.unsqueeze_(dim=0).transpose(2, 1).contiguous()

        return outputs
