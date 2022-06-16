import torch
import torch.nn as nn
import torch.nn.functional as F
from pcs.utils.torchutils import grad_reverse, initialize_weights


class Classifier(nn.Module):
    def __init__(self, cls_dim=(101, 56, 2)):
        super(Classifier, self).__init__()
        self.cls_dim = cls_dim
        self.num_class = cls_dim[0] * cls_dim[1] * cls_dim[2]

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.num_class)
        )

        initialize_weights(self.cls)

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x_out = self.cls(x)

        # group_cls, resize 11312 to (101, 56, 2)
        x_out = x_out.view(-1, *self.cls_dim)

        return x_out


class CosineClassifier(nn.Module):
    def __init__(self, cls_dim=(101, 56, 2), temp=0.05):
        super(CosineClassifier, self).__init__()
        self.cls_dim = cls_dim
        self.num_class = cls_dim[0] * cls_dim[1] * cls_dim[2]
        self.temp = temp

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.num_class)
        )

        initialize_weights(self.cls)

    def forward(self, x, reverse=False, eta=0.1):
        self.cls[0].weight.data = F.normalize(self.cls[0].weight.data, p=2, eps=1e-12, dim=1)
        self.cls[2].weight.data = F.normalize(self.cls[2].weight.data, p=2, eps=1e-12, dim=1)

        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.cls(x)
        x_out /= self.temp

        # group_cls, resize 11312 to (101, 56, 2)
        x_out = x_out.view(-1, *self.cls_dim)

        return x_out

    @torch.no_grad()
    def compute_discrepancy(self):
        self.cls[0].weight.data = F.normalize(self.cls[0].weight.data, p=2, eps=1e-12, dim=1)
        self.cls[2].weight.data = F.normalize(self.cls[2].weight.data, p=2, eps=1e-12, dim=1)

        W = self.cls[2].weight.data
        D = torch.mm(W, W.transpose(0, 1))
        D_mask = 1 - torch.eye(self.num_class).cuda()
        return torch.sum(D * D_mask).item()
