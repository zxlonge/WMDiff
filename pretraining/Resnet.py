
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


def _conv_variance_scaling_se(in_channel, out_channel, kernel_size, stride, padding, bias=False):
    weight = init.normal_(torch.empty(out_channel, in_channel, kernel_size, kernel_size), mean=0,
                          std=math.sqrt(1.0 / (in_channel * kernel_size * kernel_size)))
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    conv.weight = nn.Parameter(weight)
    return conv


def _conv_kaiming_normal(in_channel, out_channel, kernel_size, stride, padding, bias=False):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    return conv


def _conv_kaiming_uniform(in_channel, out_channel, kernel_size, stride, padding, bias=False):
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_uniform_(conv.weight, mode='fan_out', nonlinearity='relu')
    return conv


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        return _conv_variance_scaling_se(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
    else:
        return _conv_kaiming_normal(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        return _conv_variance_scaling_se(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)
    else:
        return _conv_kaiming_normal(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        return _conv_variance_scaling_se(in_channel, out_channel, kernel_size=7, stride=stride, padding=3, bias=False)
    else:
        return _conv_kaiming_normal(in_channel, out_channel, kernel_size=7, stride=stride, padding=3, bias=False)


class _bn(nn.Module):
    def __init__(self, num_features):
        super(_bn, self).__init__()
        self.bn_0 = nn.BatchNorm2d(num_features)
        self.bn_1 = nn.BatchNorm2d(num_features)
        self.bn_2 = nn.BatchNorm2d(num_features)
        self.bn_3 = nn.BatchNorm2d(num_features)
        self.bn_4 = nn.BatchNorm2d(num_features)

    def forward(self, x, bn_selection=0):
        result = 0
        if bn_selection == 0:
            result = self.bn_0(x)
        elif bn_selection == 1:
            result = self.bn_1(x)
        elif bn_selection == 2:
            result = self.bn_2(x)
        elif bn_selection == 3:
            result = self.bn_3(x)
        elif bn_selection == 4:
            result = self.bn_4(x)
        return result


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _bn(planes)
        self.downsample = downsample
        self.cbam = CBAM(planes)

    def forward(self, x, bn_selection):
        residual = x

        out = F.relu(self.bn1(self.conv1(x), bn_selection))
        out = self.bn2(self.conv2(out), bn_selection)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.cbam:
            out = self.cbam(out)
        out += residual
        out = F.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = _bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = _bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                _bn(self.expansion * planes)
            )

    def forward(self, x, bn_selection):
        out = F.relu(self.bn1(self.conv1(x), bn_selection))
        out = self.bn2(self.conv2(out), bn_selection)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = _bn(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = _bn(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, bias=False)
        self.bn3 = _bn(self.expansion * planes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, bn_selection):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, bn_selection)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out, bn_selection)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out, bn_selection)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, layers):
        super(FPN, self).__init__()
        topchannel = [2048, 1024, 512, 256, 64] if (block == Bottleneck) else [512, 256, 128, 64, 64]
        self.layers = layers  # block 块的个数，eg: resnet50->[3,4,6,3]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # input_image: gray
        # self.conv1 = _conv7x7(in_channel=1, out_channel=64, stride=2)
        self.bn1 = _bn(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.toplayer = nn.Conv2d(topchannel[0], 64, 1, 1, 0)

        self.smooth1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth4 = nn.Conv2d(64, 64, 3, 1, 1)

        self.latlayer1 = nn.Conv2d(topchannel[1], 64, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(topchannel[2], 64, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(topchannel[3], 64, 1, 1, 0)
        self.latlayer4 = nn.Conv2d(topchannel[4], 64, 1, 1, 0)
        self.gn_conv_last1 = nn.Conv2d(64, 8, (1, 1), bias=False)
        self.gn_conv_last2 = nn.Conv2d(64, 8, (1, 1), bias=False)
        self.gn_conv_last3 = nn.Conv2d(64, 8, (1, 1), bias=False)
        self.gn_conv_last4 = nn.Conv2d(64, 8, (1, 1), bias=False)
        self.gn_conv_last5 = nn.Conv2d(64, 8, (1, 1), bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, block.expansion * planes, 1, stride, bias=False),
                _bn(block.expansion * planes)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, x, bn_selection):

        features = []  # append 4layers feature
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x), bn_selection)))
        c1_ = c1
        for m in self.layer1:
            c1_ = m(c1_, bn_selection)
        c2 = c1_
        for m in self.layer2:
            c1_ = m(c1_, bn_selection)
        c3 = c1_
        for m in self.layer3:
            c1_ = m(c1_, bn_selection)
        c4 = c1_
        for m in self.layer4:
            c1_ = m(c1_, bn_selection)
        c5 = c1_

        self.p5 = self.toplayer(c5)
        self.p4 = self._upsample_add(self.p5, self.latlayer1(c4))
        self.p3 = self._upsample_add(self.p4, self.latlayer2(c3))
        self.p2 = self._upsample_add(self.p3, self.latlayer3(c2))
        self.p1 = self._upsample_add(self.p2, self.latlayer4(c1))

        self.p4 = self.smooth4(self.p4)
        self.p3 = self.smooth3(self.p3)
        self.p2 = self.smooth2(self.p2)
        self.p1 = self.smooth1(self.p1)

        self.p5 = self.gn_conv_last5(self.p5).view(self.p5.shape[0], 8, -1).mean(dim=2)
        self.p4 = self.gn_conv_last4(self.p4).view(self.p4.shape[0], 8, -1).mean(dim=2)
        self.p3 = self.gn_conv_last3(self.p3).view(self.p3.shape[0], 8, -1).mean(dim=2)
        self.p2 = self.gn_conv_last2(self.p2).view(self.p2.shape[0], 8, -1).mean(dim=2)
        self.p1 = self.gn_conv_last1(self.p1).view(self.p1.shape[0], 8, -1).mean(dim=2)  # 测试4个尺度

        for i in range(0, 5):
            layer_name = f'p{i + 1}'
            layer = getattr(self, layer_name)
            features.append(layer)

        self.y_single = torch.stack(features, dim=1)

        self.y_fusion = 0.3333 * (self.p5 + self.p4 + self.p3)  # + self.p2 + self.p1)

        return self.y_fusion, self.y_single
