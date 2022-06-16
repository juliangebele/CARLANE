import torch
import torch.nn.functional as F
from model.backbone import resnet
import numpy as np
from utils.reverse_layer import ReverseLayerF


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, backbone='18', pretrained=True, use_aux=False, cls_dim=(37, 10, 4)):
        super(Encoder, self).__init__()
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        self.model = resnet(backbone, pretrained=pretrained)

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3, padding=2, dilation=2),
                conv_bn_relu(256, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1, 1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        self.pool = torch.nn.Conv2d(512, 8, 1) if backbone in ['34', '18'] else torch.nn.Conv2d(2048, 8, 1)

    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        x2, x3, feat = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
            x4 = self.aux_header4(feat)
            x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=True)
            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)

        feat = self.pool(feat).view(-1, 1800)

        if self.use_aux:
            return feat, aux_seg

        return feat


class Classifier(torch.nn.Module):
    def __init__(self, cls_dim=(37, 10, 4)):
        super(Classifier, self).__init__()

        self.cls_dim = cls_dim
        self.total_dim = np.prod(cls_dim)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim)
        )
        initialize_weights(self.cls)

    def forward(self, feat):
        group_cls = self.cls(feat).view(-1, *self.cls_dim)
        return group_cls


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 256, 512 and 1024, Paper: 3 layer: 1080 -> 1024 -> 1024 -> 2
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(1800, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2)
        )
        initialize_weights(self.discriminator)

    def forward(self, feat, alpha):
        reverse_feature = ReverseLayerF.apply(feat, alpha)
        domain_output = self.discriminator(reverse_feature)
        return F.softmax(domain_output, dim=1)


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
