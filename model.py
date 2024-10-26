import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.densenet import densenet121


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(in_planes, in_planes // ratio, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_planes // ratio, in_planes, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(x)))
        max_out = self.fc2(self.relu1(self.fc1(x)))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class ConditionalModel(nn.Module):
    def __init__(self, config, guidance=False):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1

        y_dim = config.data.num_classes
        arch = config.model.arch
        feature_dim = config.model.feature_dim  # 6144

        self.guidance = guidance

        self.encoder_x = ResNetEncoder(arch=arch, feature_dim=feature_dim)

        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat=None, mapped_t=None):
        x = self.encoder_x(x)
        x = self.norm(x)
        if y.size() != yhat.size():
            indices = mapped_t.long()
            selected_y_0_hat = torch.zeros_like(y)
            for i in range(y.shape[0]):
                selected_y_0_hat[i, :] = yhat[i, indices[i], :]
            yhat = selected_y_0_hat
        if self.guidance:
            y = torch.cat([y, yhat], dim=-1)

        y = self.lin1(y, t)
        y = self.unetnorm1(y)

        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)

        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)

        y = F.softplus(y)
        y = self.lin4(y)

        return y


# ResNet 18 or 50 as image encoder
class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128):
        super(ResNetEncoder, self).__init__()

        self.f = []
        # print(arch)
        if arch == 'resnet50':
            backbone = resnet50(pretrained=True)
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'resnet18':
            backbone = resnet18(pretrained=True)
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'densenet121':
            backbone = densenet121(pretrained=True)
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.featdim = backbone.classifier.weight.shape[1]

        for name, module in backbone.named_children():
            # if not isinstance(module, nn.Linear):
            #    self.f.append(module)
            if name != 'fc':
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)

        # print(self.featdim)
        self.g = nn.Linear(self.featdim, feature_dim)
        # self.z = nn.Linear(feature_dim, 4)

    def forward_feature(self, x):
        feature = self.f(x)
        # x = x.mean(dim=1)

        feature = torch.flatten(feature, start_dim=1)
        feature = self.g(feature)

        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature
