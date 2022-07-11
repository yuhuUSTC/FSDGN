import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.fc1var = nn.Linear(2*64*28*28, 3)
        self.sig1var = nn.Sigmoid()

    def noise(self, mu, logvar, return_std_pre=False, return_std=False, eval=False):

        std = torch.exp(0.5*logvar)

        if return_std_pre:
            return mu, std

        std = torch.clamp(std, 0, 1)
        mu = torch.clamp(mu, -2, 2)

        if return_std:
            return mu, std

        eps = torch.randn_like(mu)

        if eval:
            return mu

        return mu + eps*std

    def forward(self, x, mix=True, return_feat=False, noise_layer=True, eval=False):

        in_size = x.size(0)

        if not noise_layer:
            out1 = self.relu1(self.mp(self.conv1(x)))
            out2 = self.relu2(self.mp(self.conv2(out1)))
        else:
            out1 = self.relu1(self.mp(self.conv1(x)))
            out2 = self.relu2(self.mp(self.conv2(out1)))

            noise1 = F.softplus(self.noise(self.conv1_1(x), self.conv1_2(x), eval=eval))
            out1_noise = self.relu1_1(self.mp(self.conv1(x) + noise1))

            noise2 = F.softplus(self.noise(self.conv2_1(out1_noise), self.conv2_2(out1_noise), eval=eval))
            out2_noise = self.relu2_1(self.mp(self.conv2(out1_noise) + noise2))

        if return_feat:
            if not noise_layer:
                return out1, out2
            else:
                return out1_noise, out2_noise
        else:
            if noise_layer:
                return out1
            else:
                return out1
