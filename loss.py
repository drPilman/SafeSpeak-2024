from torch import nn


class CapsuleLoss(nn.Module):
    def __init__(self, gpu_id, weight):
        super(CapsuleLoss, self).__init__()
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=self.weight)

        if gpu_id >= 0:
            self.weight.cuda(gpu_id)
            self.ce.cuda(gpu_id)

    def forward(self, classes, labels):
        loss_t = self.ce(classes[:, 0, :], labels)

        for i in range(classes.size(1) - 1):
            loss_t = loss_t + self.ce(classes[:, i + 1, :], labels)

        return loss_t
