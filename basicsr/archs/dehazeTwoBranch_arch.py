import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import math
from basicsr.utils import IMDB as B
from .arch_util import ResidualBlockNoBN, make_layer
import matplotlib.pyplot as plt
import numpy as np
import cv2


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, scale=1.0):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        self.scale = scale
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out) * self.scale
        out = out + x
        return out


class ASPP(nn.Module):
    def __init__(self, in_channel=3, depth=16):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(
            torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1))
        return net


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )
            self.up_convs.append(
                DeconvBlock(num_filter * (2 ** (i + 1)), num_filter * (2 ** i), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft - len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft - i - 1](ft_fusion - ft_l_list[i]) + ft_h_list[
                    len(ft_l_list) - i - 1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = F.interpolate(ft, size=ft_l_list[i].shape[-2:], mode='bilinear')
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = F.interpolate(ft_fusion, size=ft.shape[-2:], mode='bilinear')
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class Encoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter // (2 ** i), num_filter // (2 ** (i + 1)), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )
            self.down_convs.append(
                ConvBlock(num_filter // (2 ** (i + 1)), num_filter // (2 ** i), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )

    def forward(self, ft_l, ft_h_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft - len(ft_h_list) + i](ft_l)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft - i - 1](ft_fusion - ft_h_list[i]) + ft_l_list[
                    len(ft_h_list) - i - 1]

        if self.mode == 'iter2':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = F.interpolate(ft, size=ft_h_list[i].shape[-2:], mode='bilinear')
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = F.interpolate(ft_fusion, size=ft.shape[-2:], mode='bilinear')
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)



class HIN(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HIN, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)

        # if use_HIN:
        #     self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        # self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        # if self.use_HIN:
        #     out_1, out_2 = torch.chunk(out, 2, dim=1)
        #     out_1 = self.norm(out_1)
        #     feature_save(out_1,'IN')
        #     feature_save(out_2,'ID')
        #     out = torch.cat([out_1, out_2], dim=1)

        out = self.relu(out)
        out = self.relu(self.conv_2(out))
        out += self.identity(x)

        return out


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'HIN':
            return HIN(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, subnet_constructor=subnet('HIN'),
                 clamp=0.8):  ################  split_channel一般设为channel_num的一半
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out + x


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class GroupSubnetFusion(nn.Module):
    def __init__(self, channel_num, kernel=3):
        super(GroupSubnetFusion, self).__init__()
        self.SmallSA = SpatialAttention(kernel)
        self.MiddleSA = SpatialAttention(kernel)

        # self.smallUpChannel = nn.Conv2d(in_channels=channel_num // 2, out_channels=channel_num, kernel_size=kernel, padding=1, groups=2)
        self.conv = nn.Conv2d(in_channels=channel_num * 2, out_channels=channel_num, kernel_size=kernel, padding=1,
                              groups=2)

    def Itv_concat(self, middle, small):
        B, C, H, W = small.shape
        middle = middle[:, :, None, :, :]
        small = small[:, :, None, :, :]
        fuse = torch.cat([middle, small], dim=2)
        return fuse.reshape(B, 2 * C, H, W)

    def forward(self, x1, x2):
        x1 = self.MiddleSA(x1) * x1
        x2 = self.SmallSA(x2) * x2

        group_cat = self.Itv_concat(x1, x2)
        fusion = self.conv(group_cat)
        return fusion


class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up == nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_chans*2, out_chans, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=False):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat=3, kernel_size=3, reduction=1, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer1(nn.Module):
    def __init__(self, channel):
        super(CALayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size=3):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer1(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class HinBlock(nn.Module):
    def __init__(self, channel):
        super(HinBlock, self).__init__()
        self.conv = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.layers = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                    )
        self.conv_1x1 = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.IN = nn.InstanceNorm2d(channel / 2)

    def forward(self, x):
        Q = self.conv(x)
        Q1, Q2 = torch.chunk(Q, 2, dim=1)
        Q1 = self.IN(Q1)
        Q = torch.cat([Q1, Q2], dim=1)
        out = self.layers(Q)
        short = self.conv_1x1(x)
        out = out + short
        return out


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                    )
        self.conv_1x1 = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.layers(x)
        short = self.conv_1x1(x)
        out = out + short
        return out


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, num, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(Decoder_MDCBlock1, self).__init__()
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        a = num_filter
        for i in range(self.num_ft):
            b = a + 2 ** (num + i)
            self.down_convs.append(ConvBlock(a, b, kernel_size, stride, padding, bias, activation, norm=None))
            self.up_convs.append(DeconvBlock(b, a, kernel_size, stride, padding, bias, activation, norm=None))
            a = b

    def forward(self, ft_h, ft_l_list):
        ft_fusion = ft_h
        for i in range(len(ft_l_list)):
            ft = ft_fusion
            for j in range(self.num_ft - i):
                ft = self.down_convs[j](ft)
            ft = F.interpolate(ft, size=ft_l_list[i].shape[-2:], mode='bilinear')
            ft = ft - ft_l_list[i]
            for j in range(self.num_ft - i):
                ft = self.up_convs[self.num_ft - i - j - 1](ft)
            ft_fusion = F.interpolate(ft_fusion, size=ft.shape[-2:], mode='bilinear')
            ft_fusion = ft_fusion + ft

        return ft_fusion


class Encoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(Encoder_MDCBlock1, self).__init__()

        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        a = num_filter
        for i in range(self.num_ft):
            b = a - 2**(num_ft-i)
            self.up_convs.append(DeconvBlock(a, b, kernel_size, stride, padding, bias, activation, norm=None))
            self.down_convs.append(ConvBlock(b, a, kernel_size, stride, padding, bias, activation, norm=None))
            a = b

    def forward(self, ft_l, ft_h_list):
        ft_fusion = ft_l
        for i in range(len(ft_h_list)):
            ft = ft_fusion
            for j in range(self.num_ft - i):
                ft = self.up_convs[j](ft)
            ft = F.interpolate(ft, size=ft_h_list[i].shape[-2:], mode='bilinear')
            ft = ft - ft_h_list[i]
            for j in range(self.num_ft - i):
                # print(j)
                ft = self.down_convs[self.num_ft - i - j - 1](ft)
            ft_fusion = F.interpolate(ft_fusion, size=ft.shape[-2:], mode='bilinear')
            ft_fusion = ft_fusion + ft

        return ft_fusion




class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat):
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.mag(mag)
        mag_res = mag_out - mag
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y



@ARCH_REGISTRY.register()
class MPRfusion(nn.Module):
    def __init__(self, num_in_ch=3, base_channel=16, up_mode='upconv', bias=False):
        super(MPRfusion, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.layer0 = nn.Conv2d(num_in_ch, base_channel, kernel_size=3, stride=1, padding=1)
        self.layer1 = UNetConvBlock(in_chans=16, out_chans=20)
        self.layer2 = UNetConvBlock(in_chans=20, out_chans=28)
        self.layer3 = UNetConvBlock(in_chans=28, out_chans=44)
        self.layer4 = UNetConvBlock(in_chans=44, out_chans=76)
        self.layer_0 = UNetUpBlock(in_chans=20, out_chans=16, up_mode=up_mode)
        self.layer_1 = UNetUpBlock(in_chans=28, out_chans=20, up_mode=up_mode)
        self.layer_2 = UNetUpBlock(in_chans=44, out_chans=28, up_mode=up_mode)
        self.layer_3 = UNetUpBlock(in_chans=76, out_chans=44, up_mode=up_mode)

        self.layer0_ = nn.Conv2d(num_in_ch, base_channel, kernel_size=3, stride=1, padding=1)
        self.layer1_ = UNetConvBlock(in_chans=16, out_chans=20)
        self.layer2_ = UNetConvBlock(in_chans=20, out_chans=28)
        self.layer3_ = UNetConvBlock(in_chans=28, out_chans=44)
        self.layer4_ = UNetConvBlock(in_chans=44, out_chans=76)
        self.layer_0_ = UNetUpBlock(in_chans=20, out_chans=16, up_mode=up_mode)
        self.layer_1_ = UNetUpBlock(in_chans=28, out_chans=20, up_mode=up_mode)
        self.layer_2_ = UNetUpBlock(in_chans=44, out_chans=28, up_mode=up_mode)
        self.layer_3_ = UNetUpBlock(in_chans=76, out_chans=44, up_mode=up_mode)

        self.last = nn.Conv2d(base_channel, num_in_ch, kernel_size=1)

        self.fft0 = ResBlock_fft_bench(n_feat=base_channel)
        self.fft1 = ResBlock_fft_bench(n_feat=20)
        self.fft2 = ResBlock_fft_bench(n_feat=28)
        self.fft3 = ResBlock_fft_bench(n_feat=44)
        self.fft4 = ResBlock_fft_bench(n_feat=76)
        self.fft_0 = ResBlock_fft_bench(n_feat=base_channel)
        self.fft_1 = ResBlock_fft_bench(n_feat=20)
        self.fft_2 = ResBlock_fft_bench(n_feat=28)
        self.fft_3 = ResBlock_fft_bench(n_feat=44)

        self.res0 = ResBlock(base_channel)
        self.res1 = ResBlock(20)
        self.res2 = ResBlock(28)
        self.res3 = ResBlock(44)
        self.res4 = ResBlock(76)
        self.res_0 = ResBlock(base_channel)
        self.res_1 = ResBlock(20)
        self.res_2 = ResBlock(28)
        self.res_3 = ResBlock(44)

        self.res0_ = ResBlock(base_channel)
        self.res1_ = ResBlock(20)
        self.res2_ = ResBlock(28)
        self.res3_ = ResBlock(44)
        self.res4_ = ResBlock(76)
        self.res_0_ = ResBlock(base_channel)
        self.res_1_ = ResBlock(20)
        self.res_2_ = ResBlock(28)
        self.res_3_ = ResBlock(44)


        self.fusion1 = Encoder_MDCBlock1(20, 2)
        self.fusion2 = Encoder_MDCBlock1(28, 3)
        self.fusion3 = Encoder_MDCBlock1(44, 4)
        self.fusion4 = Encoder_MDCBlock1(76, 5)
        self.fusion_3 = Decoder_MDCBlock1(44, 2, 5)
        self.fusion_2 = Decoder_MDCBlock1(28, 3, 4)
        self.fusion_1 = Decoder_MDCBlock1(20, 4, 3)
        self.fusion_0 = Decoder_MDCBlock1(base_channel, 5, 2)

        self.fusion1_ = Encoder_MDCBlock1(20, 2)
        self.fusion2_ = Encoder_MDCBlock1(28, 3)
        self.fusion3_ = Encoder_MDCBlock1(44, 4)
        self.fusion4_ = Encoder_MDCBlock1(76, 5)
        self.fusion_3_ = Decoder_MDCBlock1(44, 2, 5)
        self.fusion_2_ = Decoder_MDCBlock1(28, 3, 4)
        self.fusion_1_ = Decoder_MDCBlock1(20, 4, 3)
        self.fusion_0_ = Decoder_MDCBlock1(base_channel, 5, 2)

        self.sam = SAM(base_channel, kernel_size=1)
        #self.concat = conv(base_channel * 2, base_channel, kernel_size=3)

    def forward(self, x):
        xcopy = x
        blocks = []
        x = self.layer0(x)
        x0 = self.res0(x)
        x0 = self.fft0(x0)
        blocks.append(x0)

        x1 = self.layer1(x0)
        x1 = self.fusion1(x1, blocks)
        x1 = self.res1(x1)
        x1 = self.fft1(x1)
        blocks.append(x1)

        x2 = self.layer2(x1)
        x2 = self.fusion2(x2, blocks)
        x2 = self.res2(x2)
        x2 = self.fft2(x2)
        blocks.append(x2)

        x3 = self.layer3(x2)
        x3 = self.fusion3(x3, blocks)
        x3 = self.res3(x3)
        x3 = self.fft3(x3)
        blocks.append(x3)

        x4 = self.layer4(x3)
        x4 = self.fusion4(x4, blocks)
        x4 = self.res4(x4)
        x4 = self.fft4(x4)

        blocks_up = [x4]
        x_3 = self.layer_3(x4, blocks[-0 - 1])
        x_3 = self.res_3(x_3)
        x_3 = self.fft_3(x_3)
        x_3 = self.fusion_3(x_3, blocks_up)
        blocks_up.append(x_3)

        x_2 = self.layer_2(x_3, blocks[-1 - 1])
        x_2 = self.res_2(x_2)
        x_2 = self.fft_2(x_2)
        x_2 = self.fusion_2(x_2, blocks_up)
        blocks_up.append(x_2)

        x_1 = self.layer_1(x_2, blocks[-2 - 1])
        x_1 = self.res_1(x_1)
        x_1 = self.fft_1(x_1)
        x_1 = self.fusion_1(x_1, blocks_up)
        blocks_up.append(x_1)

        x_0 = self.layer_0(x_1, blocks[-3 - 1])
        x_0 = self.res_0(x_0)
        x_0 = self.fft_0(x_0)
        x_0 = self.fusion_0(x_0, blocks_up)


        x2_samfeats, stage1_output = self.sam(x_0, xcopy)

        blocks1 = []
        y = self.layer0_(xcopy)
        #y = self.concat(torch.cat([y, x2_samfeats], 1))

        y0 = self.res0_(y)
        #y0 = y0 + self.csff_enc0(x0) + self.csff_dec0(x_0)
        blocks1.append(y0)

        y1 = self.layer1_(y0)
        y1 = self.fusion1_(y1, blocks1)
        y1 = self.res1_(y1)
        #y1 = y1 + self.csff_enc1(x1) + self.csff_dec1(x_1)
        blocks1.append(y1)

        y2 = self.layer2_(y1)
        y2 = self.fusion2_(y2, blocks1)
        y2 = self.res2_(y2)
        #y2 = y2 + self.csff_enc2(x2) + self.csff_dec2(x_2)
        blocks1.append(y2)


        y3 = self.layer3_(y2)
        y3 = self.fusion3_(y3, blocks1)
        y3 = self.res3_(y3)
        #y3 = y3 + self.csff_enc3(x3) + self.csff_dec3(x_3)
        blocks1.append(y3)


        y4 = self.layer4_(y3)
        y4 = self.fusion4_(y4, blocks1)
        y4 = self.res4_(y4)

        blocks1_up = [y4]
        y_3 = self.layer_3_(y4, blocks1[-0 - 1])
        y_3 = self.res_3_(y_3)
        y_3 = self.fusion_3_(y_3, blocks1_up)
        blocks1_up.append(y_3)


        y_2 = self.layer_2_(y_3, blocks1[-1 - 1])
        y_2 = self.res_2_(y_2)
        y_2 = self.fusion_2_(y_2, blocks1_up)
        blocks1_up.append(y_2)


        y_1 = self.layer_1_(y_2, blocks1[-2 - 1])
        y_1 = self.res_1_(y_1)
        y_1 = self.fusion_1_(y_1, blocks1_up)
        blocks1_up.append(y_1)


        y_0 = self.layer_0_(y_1, blocks1[-3 - 1])
        y_0 = self.res_0_(y_0)
        y_0 = self.fusion_0_(y_0, blocks1_up)


        # y_0 = self.INN1(y_0)

        output = self.last(y_0)

        output = torch.clamp(output, 0, 1)
        stage1_output = torch.clamp(stage1_output, 0, 1)

        a = stage1_output[0, :]
        pred = a.permute(1, 2, 0)
        heatmap = pred.cpu().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap1 = cv2.normalize(heatmap, 0, 255.0, norm_type=cv2.NORM_MINMAX)
        heatmap1 = cv2.cvtColor(heatmap1, cv2.COLOR_BGR2RGB)
        size = (620, 460)
        heatmap1 = cv2.resize(heatmap1, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(r'D:\global.png', heatmap1)

        a = output[0, :]
        pred = a.permute(1, 2, 0)
        heatmap = pred.cpu().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap2 = cv2.normalize(heatmap, 0, 255.0, norm_type=cv2.NORM_MINMAX)
        heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_BGR2RGB)
        size = (620, 460)
        heatmap2 = cv2.resize(heatmap2, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(r'D:\local.png', heatmap2)

        return output, stage1_output



    '''
        x_ave = F.adaptive_avg_pool2d(x0, (1, 1))
        cos_sim = (F.normalize(x_ave, dim=1) * F.normalize(x0, dim=1)).sum(1).cpu().numpy()
        plt.figure("1")
        arr = cos_sim.flatten()
        #cos_sim = cos_sim.view(N, -1)
        plt.hist(arr, bins=256, density=1, stacked=True, facecolor='green', alpha=0.75)
        plt.show()
        
        
        
        a = x0[0, :]
        pred = a.permute(1, 2, 0)
        pred = pred.cpu().numpy()
        heatmap = np.mean(pred, axis=2)
        heatmap1 = cv2.normalize(heatmap, 0, 1, norm_type=cv2.NORM_MINMAX)
        sc = plt.imshow(heatmap1)
        sc.set_cmap('hot')  # 这里可以设置多种模式
        plt.colorbar()  # 显示色度条
        plt.axis('off')  # 关掉坐标轴为 off
        #plt.title('Hazy')  # 图像题目
        plt.show()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap1 = cv2.normalize(heatmap, 0, 255.0, norm_type=cv2.NORM_MINMAX)
        heatmap2 = cv2.applyColorMap(np.uint8(heatmap * 255.0), cv2.COLORMAP_JET)
        cv2.imwrite(r'D:\visualization\a.png', heatmap1)
        cv2.imwrite(r'D:\visualization\a1.png', heatmap2)
    '''
