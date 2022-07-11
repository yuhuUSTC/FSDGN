import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer
from mmedit.models.common import (PixelShufflePack, flow_warp)
from basicsr.archs.spynet import SPyNet, ResidualBlocksWithInputConv
from positional_encodings import PositionalEncodingPermute3D


# Transformer
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, num_feat, feat_size, fn):
        super().__init__()
        self.norm = nn.LayerNorm([num_feat, feat_size, feat_size])
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FeedForward(nn.Module):
    def __init__(self, num_feat):
        super().__init__()

        self.backward_resblocks = ResidualBlocksWithInputConv(num_feat + 3, num_feat, num_blocks=30)
        self.forward_resblocks = ResidualBlocksWithInputConv(num_feat + 3, num_feat, num_blocks=30)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, lrs=None, flows=None):
        b, t, c, h, w = x.shape
        x1 = torch.cat([x[:, 1:, :, :, :], x[:, -1, :, :, :].unsqueeze(1)], dim=1)  # [B, 3, 3, 64, 64]
        flow1 = flows[1].contiguous().view(-1, 2, h, w).permute(0, 2, 3, 1)  # [B*3, 64, 64, 2]
        x1 = flow_warp(x1.view(-1, c, h, w), flow1)  # [B*3, 3, 64, 64]
        x1 = torch.cat([lrs.view(b * t, -1, h, w), x1], dim=1)  # [B*5, 67, 64, 64]
        x1 = self.backward_resblocks(x1)  # [B*5, 64, 64, 64]

        x2 = torch.cat([x[:, 0, :, :, :].unsqueeze(1), x[:, :-1, :, :, :]], dim=1)  # [B, 5, 64, 64, 64]
        flow2 = flows[0].contiguous().view(-1, 2, h, w).permute(0, 2, 3, 1)  # [B*5, 64, 64, 2]
        x2 = flow_warp(x2.view(-1, c, h, w), flow2)  # [B*5, 64, 64, 64]
        x2 = torch.cat([lrs.view(b * t, -1, h, w), x2], dim=1)  # [B*5, 67, 64, 64]
        x2 = self.forward_resblocks(x2)  # [B*5, 64, 64, 64]

        # fusion the backward and forward features
        out = torch.cat([x1, x2], dim=1)  # [B*5, 128, 64, 64]
        out = self.lrelu(self.fusion(out))  # [B*5, 64, 64, 64]
        out = out.view(x.shape)  # [B, 5, 64, 64, 64]

        return out


class MatmulNet(nn.Module):
    def __init__(self) -> None:
        super(MatmulNet, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, y)
        return x


class globalAttention(nn.Module):
    def __init__(self, num_feat=64, patch_size=8, heads=1):
        super(globalAttention, self).__init__()
        self.heads = heads
        self.dim = patch_size ** 2 * num_feat
        self.hidden_dim = self.dim // heads
        self.num_patch = (64 // patch_size) ** 2

        self.to_q = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_k = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_v = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)
        self.patch2feat = torch.nn.Fold(output_size=(64, 64), kernel_size=patch_size, padding=0, stride=patch_size)

    def forward(self, x):
        b, t, c, h, w = x.shape  # B, 3, 3, 64, 64
        H, D = self.heads, self.dim
        n, d = self.num_patch, self.hidden_dim

        q = self.to_q(x.view(-1, c, h, w))  # [B*5, 64, 64, 64]
        k = self.to_k(x.view(-1, c, h, w))  # [B*5, 64, 64, 64]
        v = self.to_v(x.view(-1, c, h, w))  # [B*5, 64, 64, 64]

        unfold_q = self.feat2patch(q)  # [B*5, 8*8*64, 8*8]
        unfold_k = self.feat2patch(k)  # [B*5, 8*8*64, 8*8]
        unfold_v = self.feat2patch(v)  # [B*5, 8*8*64, 8*8]

        unfold_q = unfold_q.view(b, t, H, d, n)  # [B, 5, H, 8*8*64/H, 8*8]
        unfold_k = unfold_k.view(b, t, H, d, n)  # [B, 5, H, 8*8*64/H, 8*8]
        unfold_v = unfold_v.view(b, t, H, d, n)  # [B, 5, H, 8*8*64/H, 8*8]

        unfold_q = unfold_q.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 8*8]
        unfold_k = unfold_k.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 8*8]
        unfold_v = unfold_v.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*64/H, 5, 8*8]

        unfold_q = unfold_q.view(b, H, d, t * n)  # [B, H, 8*8*64/H, 5*8*8]
        unfold_k = unfold_k.view(b, H, d, t * n)  # [B, H, 8*8*64/H, 5*8*8]
        unfold_v = unfold_v.view(b, H, d, t * n)  # [B, H, 8*8*64/H, 5*8*8]

        attn = torch.matmul(unfold_q.transpose(2, 3), unfold_k)  # [B, H, 5*8*8, 5*8*8]
        attn = attn * (d ** (-0.5))  # [B, H, 5*8*8, 5*8*8]
        attn = F.softmax(attn, dim=-1)  # [B, H, 5*8*8, 5*8*8]

        attn_x = torch.matmul(attn, unfold_v.transpose(2, 3))  # [B, H, 5*8*8, 8*8*64/H]
        attn_x = attn_x.view(b, H, t, n, d)  # [B, H, 5, 8*8, 8*8*64/H]
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()  # [B, 5, H, 8*8*64/H, 8*8]
        attn_x = attn_x.view(b * t, D, n)  # [B*5, 8*8*64, 8*8]
        feat = self.patch2feat(attn_x)  # [B*5, 64, 64, 64]

        out = self.conv(feat).view(x.shape)  # [B, 5, 64, 64, 64]
        out += x  # [B, 5, 64, 64, 64]

        return out


class Transformer(nn.Module):
    def __init__(self, num_feat, feat_size, depth, patch_size, heads):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(num_feat, feat_size, globalAttention(num_feat, patch_size, heads))),
                Residual(PreNorm(num_feat, feat_size, FeedForward(num_feat)))
            ]))

    def forward(self, x, lrs=None, flows=None):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x, lrs=lrs, flows=flows)
        return x


class Flow_spynet(nn.Module):
    def __init__(self, spynet_pretrained=None):
        super(Flow_spynet, self).__init__()
        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def forward(self, lrs):
        """Compute optical flow using SPyNet for feature warping.
        Note that if the input is an mirror-extended sequence, 'flows_forward' is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, ('The height and width of inputs should be at least 64, 'f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        lrs_1 = torch.cat([lrs[:, 0, :, :, :].unsqueeze(1), lrs], dim=1).reshape(-1, c, h, w)  # [b*6, 3, 64, 64]
        lrs_2 = torch.cat([lrs, lrs[:, t - 1, :, :, :].unsqueeze(1)], dim=1).reshape(-1, c, h, w)  # [b*6, 3, 64, 64]

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t + 1, 2, h, w)  # [b, 6, 2, 64, 64]
        flows_backward = flows_backward[:, 1:, :, :, :]  # [b, 5, 2, 64, 64]

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t + 1, 2, h, w)  # [b, 6, 2, 64, 64]
            flows_forward = flows_forward[:, :-1, :, :, :]  # [b, 5, 2, 64, 64]

        return flows_forward, flows_backward


@ARCH_REGISTRY.register()
class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 num_frame=3,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 hr_in=False,
                 depth=10,
                 heads=1,
                 patch_size=8,
                 spynet_pretrained=None
                 ):
        super(EDVR, self).__init__()

        self.hr_in = hr_in
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # Optical flow
        self.spynet = Flow_spynet(spynet_pretrained)

        # Transformer
        self.pos_embedding = PositionalEncodingPermute3D(num_frame)
        self.transformer = Transformer(num_feat, num_feat, depth, patch_size, heads)

        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.downsample = nn.Sequential(self.conv_l2_1, self.conv_l3_1)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        b, t, c, h, w = x.size()

        # extract features for each frame
        feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat = self.feature_extraction(feat)

        x1 = F.interpolate(x.view(-1, c, h, w), size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        x1 = x1.view(b, t, -1, h // 4, w // 4)

        if self.hr_in:
            h, w = h // 4, w // 4

        # Downsample
        # feat = self.downsample(feat).view(b, t, -1, h, w)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        feat = feat.view(b, t, -1, h, w)

        # compute optical flow
        flows = self.spynet(x1)  # [4 , 3 , 2, 64, 64] x 2

        # transformer
        feat = feat + self.pos_embedding(feat)
        tr_feat = self.transformer(feat, x1, flows)
        feat = tr_feat.view(b * t, -1, h, w)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        # residual

        out = out.view(b, t, c, h * 4, w * 4)  # [B, 3, 3, 256, 256]
        out += x

        return out




import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import DCNv2Pack, ResidualBlockNoBN, make_layer


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):

        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        self.offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dcn_pack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        offset = torch.cat([nbr_feat_l, ref_feat_l], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        feat = self.lrelu(self.dcn_pack(nbr_feat_l, offset))

        # Cascading
        offset = torch.cat([feat, ref_feat_l], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class PredeblurModule(nn.Module):
    """Pre-dublur module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # generate feature pyramid
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList([ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


@ARCH_REGISTRY.register()
class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 num_frame=3,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 hr_in=False,
                 center_frame_idx=1,
                 with_predeblur=False,
                 with_tsa=False):
        super(EDVR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.downsample = nn.Sequential(self.conv_l2_1, self.conv_l2_2, self.conv_l3_1, self.conv_l3_2)
        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        # if self.hr_in:
        #     assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        # else:
        #     assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        if self.hr_in:
            h, w = h // 4, w // 4

        #feat_l1 = self.feature_extraction(feat_l1)

        # Downsample
        feat_l1 = self.downsample(feat_l1)
        feat_l1 = feat_l1.view(b, t, -1, h, w)

        # PCD alignment
        ref_feat_l = feat_l1[:, self.center_frame_idx, :, :, :].clone()  # reference feature list
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = feat_l1[:, i, :, :, :].clone()  # neighboring feature list
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out








import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import math
from typing import Optional
from torch import nn, Tensor
from .arch_util import  ResidualBlockNoBN, make_layer
from basicsr.utils.normalization import InstanceL2Norm
from basicsr.utils.multihead_attention import MultiheadAttention


@ARCH_REGISTRY.register()
class EDVR(nn.Module):
    def __init__(self, n_head=1, num_layers=1):
        super().__init__()
        num_feat = 3
        multihead_attn = MultiheadAttention(feature_dim=num_feat, n_head=n_head, key_feature_dim=3)
        # FFN_conv = nn.Conv2d()  # do not use feed-forward network
        self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=num_feat, num_encoder_layers=num_layers)
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=num_feat, num_decoder_layers=num_layers)

        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.downsample = nn.Sequential(self.conv_l2_1, self.conv_l2_2, self.conv_l3_1, self.conv_l3_2)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        train_imgs = x.lq
        mask = x.gt - x.lq

        b, t, c, h, w = train_imgs.size()
        train_imgs = self.lrelu(self.conv_first(train_imgs.view(-1, c, h, w)))
        mask = self.lrelu(self.conv_first(mask.view(-1, c, h, w)))
        h, w = h // 4, w // 4

        train_imgs = self.downsample(train_imgs)
        mask = self.downsample(mask)
        train_imgs = train_imgs.view(b, t, -1, h, w)
        mask = mask.view(b, t, -1, h, w)

        test_imgs = train_imgs[:, 1, :, :, :].unsqueeze(1)

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        # Extract backbone features
        #train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        #test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))
        # Classification features
        #train_feat_clf = self.get_backbone_clf_feat(train_feat)
        #test_feat_clf = self.get_backbone_clf_feat(test_feat)
        #train_feat = train_feat_clf.reshape(num_img_train, -1, *train_feat_clf.shape[-3:])
        #test_feat = test_feat_clf.reshape(num_img_test, -1, *test_feat_clf.shape[-3:])
        #num_sequences = train_feat.shape[1]
        #if train_feat.dim() == 5:
        #    train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        #if test_feat.dim() == 5:
        #    test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])
        # Extract features
        #train_feat = self.extractor(train_feat)
        #train_feat = train_feat.reshape(-1, num_sequences, *train_feat.shape[-3:])
        #test_feat = self.extractor(test_feat)
        #test_feat = test_feat.reshape(-1, num_sequences, *test_feat.shape[-3:])
        #num_img_train = train_feat.shape[0]
        #num_img_test = test_feat.shape[0]

        ## encoder
        encoded_memory = self.encoder(train_imgs, pos=None)     # Len_2, Batch, Dim

        ## decoder
        #for i in range(num_img_train):
        #    _, cur_encoded_feat = self.decoder(train_imgs[i, ...].unsqueeze(0), memory=encoded_memory, pos=mask[i, ...].unsqueeze(0), query_pos=None)
        #    if i == 0:
        #        encoded_feat = cur_encoded_feat
        #    else:
        #        encoded_feat = torch.cat((encoded_feat, cur_encoded_feat), 0)
        #test_imgs = test_imgs.permute(1, 0, 2, 3, 4)
        #mask = mask.permute(1, 0, 2, 3, 4)
        num_img_test = test_imgs.shape[1]

        for i in range(num_img_test):
            cur_decoded_feat = self.decoder(test_imgs, memory=encoded_memory, pos=mask, query_pos=None)
            if i == 0:
                decoded_feat = cur_decoded_feat
            else:
                decoded_feat = torch.cat((decoded_feat, cur_decoded_feat.unsqueeze(0)), 0)
        decoded_feat = decoded_feat.squeeze(1)
        decoded_feat = self.lrelu(self.pixel_shuffle(self.upconv1(decoded_feat)))
        decoded_feat = self.lrelu(self.pixel_shuffle(self.upconv2(decoded_feat)))
        decoded_feat = self.lrelu(self.conv_hr(decoded_feat))
        return decoded_feat



class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def instance_norm(self, src, input_shape):
        batch, t, dim, h, w = input_shape
        # Normlization
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(batch, t, dim, h, w)
        return src

    def forward(self, src, input_shape, pos: Optional[Tensor] = None):
        # query = key = value = src
        query = src
        key = src
        value = src

        # self-attention
        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2
        src = self.instance_norm(src, input_shape)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=3, num_encoder_layers=1, activation="relu"):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, pos: Optional[Tensor] = None):
        assert src.dim() == 5, 'Expect 5 dimensional inputs'

        src_shape = src.shape
        #num_imgs, batch, dim, h, w = src.shape
        #src = src.reshape(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        #src = src.reshape(-1, batch, dim)

        #if pos is not None:
        #    pos = pos.view(num_imgs, batch, 1, -1).permute(0, 3, 1, 2)
        #    pos = pos.reshape(-1, batch, 1)



        for layer in self.layers:
            output = layer(src, input_shape=src_shape, pos=pos)

        # [L,B,D] -> [B,D,L]
        #output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        #output_feat = output_feat.reshape(-1, dim, h, w)
        #return output, output_feat
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=3)

        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor * pos

    def instance_norm(self, src, input_shape):
        batch, t, dim, h, w = input_shape
        # Normlization
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(batch, t, dim, h, w)
        return src

    def forward(self, tgt, memory, input_shape, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        # self-attention
        query = tgt
        key = tgt
        value = tgt

        tgt2 = self.self_attn(query=query, key=key, value=value)
        tgt = tgt + tgt2
        tgt = self.instance_norm(tgt, input_shape)

        mask = self.cross_attn(query=tgt, key=memory, value=pos)
        tgt2 = tgt * mask
        tgt2 = self.instance_norm(tgt2, input_shape)

        tgt3 = self.cross_attn(query=tgt, key=memory, value=memory * pos)
        tgt4 = tgt + tgt3
        tgt4 = self.instance_norm(tgt4, input_shape)

        tgt = tgt2 + tgt4
        tgt = self.instance_norm(tgt, input_shape)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_decoder_layers=6, activation="relu"):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        # self.post1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        # self.activation = _get_activation_fn(activation)
        # self.post2 = nn.Conv2d(d_model, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, tgt, memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        assert tgt.dim() == 5, 'Expect 5 dimensional inputs'
        tgt_shape = tgt.shape
        batch, t, dim, h, w = tgt.shape
        #tgt = tgt.view(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        #tgt = tgt.reshape(-1, batch, dim)

        #if pos is not None:
        #    num_pos, batch, dim, h, w = pos.shape
        #    pos = pos.view(num_pos, batch, dim, -1).permute(0, 3, 1, 2)
        #    pos = pos.reshape(-1, batch, dim)


        for layer in self.layers:
            output = layer(tgt, memory, input_shape=tgt_shape, pos=pos, query_pos=query_pos)

        # [L,B,D] -> [B,D,L]
        #output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        #output_feat = output_feat.reshape(-1, dim, h, w)
        # output = self.post2(self.activation(self.post1(output)))
        return output


def _get_clones(module, N):
    # return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    return nn.ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")













import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from .arch_util import to_2tuple, trunc_normal_
#from timm.models.layers import DropPath, trunc_normal_

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import DCNv2Pack, ResidualBlockNoBN, make_layer
from basicsr.utils import get_root_logger
from mmcv.runner import load_checkpoint


from functools import reduce, lru_cache
from operator import mul
from einops import rearrange

from .arch_util import  ResidualBlockNoBN, make_layer
#from basicsr.archs.spynet import SPyNet, ResidualBlocksWithInputConv

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)





class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# cache each stage results
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


@ARCH_REGISTRY.register()
class EDVR(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 num_frame=3,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 hr_in=False,
                 depth=10,
                 heads=1,
                 spynet_pretrained=None,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(3, 4, 4),
                 in_chans=3,
                 embed_dim=12,
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 3, 6, 6],
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False
                 ):
        super(EDVR, self).__init__()
        self.hr_in = hr_in
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv = nn.Conv3d(num_in_ch, embed_dim, kernel_size=(3, 4, 4), stride=(3, 4, 4))
        self.deconv = nn.Conv3d(embed_dim, 144, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                #dim=int(embed_dim * 2 ** i_layer),
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                #downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        #self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features = embed_dim
        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self._freeze_stages()

        # Downsample
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.downsample = nn.Sequential(self.conv_l2_1, self.conv_l3_1)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_in_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1, self.patch_size[0], 1, 1) / self.patch_size[0]
        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
                pass
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        b, t, c, h, w = x.size()                        # 1,3,3,256,256
        x = self.conv(x)
        x = self.pos_drop(x)                            # 1,12,1,64,64

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')      # 1,12,1,64,64
        x = self.deconv(x)                              # 1,144,1,64,64
        x = x.view(b, t, c, h, w)
        return x                                        # 1,3,3,256,256

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(EDVR, self).train(mode)
        self._freeze_stages()
















import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import math
import numpy as np
from typing import Optional
from torch import nn, Tensor
from .arch_util import ResidualBlockNoBN, make_layer
from basicsr.utils.normalization import InstanceL2Norm
from basicsr.utils.multihead_attention import MultiheadAttention
#from basicsr.archs import build_network
#from mmedit.models.common import (PixelShufflePack, flow_warp)
#from basicsr.archs.spynet import SPyNet, ResidualBlocksWithInputConv
from basicsr.utils import get_root_logger, imwrite, tensor2img
import cv2
@ARCH_REGISTRY.register()
class EDVR(nn.Module):
    def __init__(self, n_head=1, num_layers=1, num_frame=3):
        super().__init__()
        num_feat = 3
        num_extract_block = 3
        self.num_frame = num_frame
        num_block = 3
        multihead_attn = MultiheadAttention(feature_dim=num_feat, n_head=n_head, key_feature_dim=3)
        # FFN_conv = nn.Conv2d()  # do not use feed-forward network
        self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=num_feat, num_encoder_layers=num_layers)
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=num_feat, num_decoder_layers=num_layers)

        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        #self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.predeblur = PredeblurModule(num_feat=num_feat, num_block=num_block, num_extract_block=num_extract_block)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.downsample = nn.Sequential(self.conv_l2_1, self.conv_l2_2, self.conv_l3_1, self.conv_l3_2)
        #self.feedforward = FeedForward(num_feat=num_feat)
        #self.spynet = Flow_spynet(spynet_pretrained=True)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.rcab = RCAB()
        #self.prenet = build_network(pretrain_network_g)
        #self.load_network(self.net_g, pretrain_network_g, self.opt['path'].get('strict_load_g', True), param_key)
        #self.prenet.eval()
    def forward(self, x):
        train_imgs = x.pre
        mask = x.pre - x.lq
        # mask1 = mask[:, :, 0, :, :] + mask[:, :, 1, :, :] + mask[:, :, 2, :, :]
        # mask[:, :, 0, :, :] = x.T
        # mask[:, :, 1, :, :] = mask1
        # mask[:, :, 2, :, :] = mask1
        center_frame = self.num_frame // 2
        b, t, c, h, w = train_imgs.size()
        # x1 = F.interpolate(train_imgs.view(-1, c, h1, w1), size=(h1 // 4, w1 // 4), mode='bilinear', align_corners=False)
        # x1 = x1.view(b, t, -1, h1 // 4, w1 // 4)
        train_imgs = self.lrelu(self.conv_first(train_imgs.view(-1, c, h, w)))
        train_imgs = self.rcab(train_imgs)
        mask = self.lrelu(self.conv_first(mask.view(-1, c, h, w)))

        train_imgs = self.downsample(train_imgs)
        # train_imgs1 = self.predeblur(train_imgs)
        train_imgs = train_imgs.view(b, t, -1, h // 4, w // 4)

        # mask = train_imgs - train_imgs0
        mask = self.downsample(mask)
        mask = mask.view(b, t, -1, h // 4, w // 4)

        ## encoder
        encoded_memory = self.encoder(train_imgs, pos=None)  # Len_2, Batch, Dim

        ## decoder
        num_img_test = train_imgs.shape[1]

        for i in range(1):
            cur_decoded_feat = self.decoder(train_imgs[:, 1, :, :, :].unsqueeze(1), memory=encoded_memory, pos=mask,
                                            query_pos=None)
            if i == 0:
                decoded_feat = cur_decoded_feat
            else:
                decoded_feat = torch.cat((decoded_feat, cur_decoded_feat), 1)

        #flows = self.spynet(train_imgs)  # [4 , 3 , 2, 64, 64] x 2
        #tr_feat = self.feedforward(decoded_feat, train_imgs, flows)
        #decoded_feat = tr_feat.view(b * t, -1, h // 4, w // 4)
        decoded_feat = decoded_feat.squeeze(1)
        # train_imgs1 = self.lrelu(self.pixel_shuffle(self.upconv1(train_imgs1)))
        # train_imgs1 = self.lrelu(self.pixel_shuffle(self.upconv2(train_imgs1)))
        # train_imgs1 = self.lrelu(self.conv_hr(train_imgs1))
        # train_imgs1 = train_imgs1.view(b, t, -1, h, w)
        # train_imgs1 = train_imgs1 + x.lq

        decoded_feat = self.lrelu(self.pixel_shuffle(self.upconv1(decoded_feat)))
        decoded_feat = self.lrelu(self.pixel_shuffle(self.upconv2(decoded_feat)))
        decoded_feat = self.lrelu(self.conv_hr(decoded_feat))
        decoded_feat = decoded_feat + x.lq[:, 1, :, :, :]

        # mask1 = self.lrelu(self.pixel_shuffle(self.upconv1(mask1)))
        # mask1 = self.lrelu(self.pixel_shuffle(self.upconv2(mask1)))
        # mask1 = self.lrelu(self.conv_hr(mask1))
        # mask1 = mask1.view(b, t, -1, h1, w1)

        # return decoded_feat, mask1, train_imgs1
        return decoded_feat

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
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class PredeblurModule(nn.Module):
    def __init__(self, num_feat=3, num_block=3, num_extract_block=3):
        super(PredeblurModule, self).__init__()
        self.num_block = num_block
        self.block = PDblock(num_feat=num_feat, num_extract_block=num_extract_block)
    def forward(self, x):
        mask = torch.zeros_like(x)
        X = []
        Mask = []
        for i in range(self.num_block):
            x, mask =self.block(x, mask)
            X.append(x)
            Mask.append(mask)
        return x, mask



class PDblock(nn.Module):
    def __init__(self, num_feat=3, num_extract_block=3):
        super(PDblock, self).__init__()
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=6)
        self.conv_1 = nn.Conv2d(6, num_feat, 3, 1, 1)
        self.conv_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_3 = nn.Conv2d(6, num_feat, 3, 1, 1)
        self.conv_4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_5 = nn.Conv2d(6, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.feature_extraction(x)
        confidence = self.lrelu(self.conv_1(x))
        confidence = torch.sigmoid(self.conv_2(confidence))
        Feat = self.conv_5(x)
        train_imgs = self.lrelu(self.conv_3(x))
        train_imgs = self.conv_4(train_imgs)

        return train_imgs, confidence



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def instance_norm(self, src, input_shape):
        batch, t, dim, h, w = input_shape
        # Normlization
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(batch, t, dim, h, w)
        return src

    def forward(self, src, input_shape, pos: Optional[Tensor] = None):
        # query = key = value = src
        query = src
        key = src
        value = src

        # self-attention
        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2
        src = self.instance_norm(src, input_shape)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=3, num_encoder_layers=1, activation="relu", norm_layer=nn.LayerNorm ,mlp_ratio=4.):
        super().__init__()
        dim = 3
        encoder_layer = TransformerEncoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)


    def forward(self, src, pos: Optional[Tensor] = None):
        assert src.dim() == 5, 'Expect 5 dimensional inputs'

        src_shape = src.shape


        #depths = [2, 2, 6, 2]
        #drop_path_rate = 0.2
        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        #for i_layer in range(4):
        #    a = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
        for layer in self.layers:
            output = layer(src, input_shape=src_shape, pos=pos)
            output = output + self.drop_path(self.mlp(self.norm2(output.permute(0, 1, 3, 4, 2)))).permute(0, 1, 4, 2, 3)
        # [L,B,D] -> [B,D,L]
        #output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        #output_feat = output_feat.reshape(-1, dim, h, w)
        #return output, output_feat
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=3)

        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor * pos

    def instance_norm(self, src, input_shape):
        batch, t, dim, h, w = input_shape
        # Normlization
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(batch, t, dim, h, w)
        return src

    def forward(self, tgt, memory, input_shape, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        # self-attention
        query = tgt
        key = tgt
        value = tgt

        tgt2 = self.self_attn(query=query, key=key, value=value)
        tgt = tgt + tgt2
        tgt = self.instance_norm(tgt, input_shape)

        mask = self.cross_attn(query=tgt, key=memory, value=pos)
        tgt2 = tgt * mask
        tgt2 = self.instance_norm(tgt2, input_shape)

        tgt3 = self.cross_attn(query=tgt, key=memory, value=memory * pos)
        tgt4 = tgt + tgt3
        tgt4 = self.instance_norm(tgt4, input_shape)

        tgt = tgt2 + tgt4
        tgt = self.instance_norm(tgt, input_shape)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_decoder_layers=6, activation="relu", norm_layer=nn.LayerNorm, mlp_ratio=4.):
        super().__init__()
        dim = 3
        decoder_layer = TransformerDecoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        # self.post1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        # self.activation = _get_activation_fn(activation)
        # self.post2 = nn.Conv2d(d_model, 1, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm_layer(dim)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, tgt, memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        assert tgt.dim() == 5, 'Expect 5 dimensional inputs'
        tgt_shape = tgt.shape
        batch, t, dim, h, w = tgt.shape
        #tgt = tgt.view(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        #tgt = tgt.reshape(-1, batch, dim)

        #if pos is not None:
        #    num_pos, batch, dim, h, w = pos.shape
        #    pos = pos.view(num_pos, batch, dim, -1).permute(0, 3, 1, 2)
        #    pos = pos.reshape(-1, batch, dim)


        for layer in self.layers:
            output = layer(tgt, memory, input_shape=tgt_shape, pos=pos, query_pos=query_pos)
            output = output + self.drop_path(self.mlp(self.norm2(output.permute(0, 1, 3, 4, 2)))).permute(0, 1, 4, 2, 3)
        # [L,B,D] -> [B,D,L]
        #output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        #output_feat = output_feat.reshape(-1, dim, h, w)
        # output = self.post2(self.activation(self.post1(output)))
        return output


def _get_clones(module, N):
    # return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    return nn.ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



