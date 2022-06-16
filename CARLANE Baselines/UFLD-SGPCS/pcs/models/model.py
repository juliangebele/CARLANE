import torch
from pcs.models.backbone import ResNet
from pcs.utils.torchutils import initialize_weights


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


class Net(torch.nn.Module):
    def __init__(self, backbone='18', cls_dim=(101, 56, 2), use_aux=False, pretrained=True):
        super(Net, self).__init__()
        self.backbone = backbone
        # cls_dim = (griding_num, len(row_anchor), num_lanes)
        self.cls_dim = cls_dim
        self.use_aux = use_aux

        # input : n*c*h*w
        # output: (w+1) * sample_rows * num_lanes
        self.model = ResNet(backbone, pretrained=pretrained)

        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if self.backbone in ['34', '18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if self.backbone in ['34', '18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if self.backbone in ['34', '18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3, padding=2, dilation=2),
                conv_bn_relu(256, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                torch.nn.Conv2d(128, self.cls_dim[-1] + 1, 1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        self.pool = torch.nn.Conv2d(512, 8, 1) if self.backbone in ['34', '18'] else torch.nn.Conv2d(2048, 8, 1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * num_lanes
        # 100+1 * 56 * 2

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
        else:
            aux_seg = None

        out = self.pool(feat).view(-1, 1800)

        if self.use_aux:
            return out, aux_seg

        return out
