import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import pdb


class MultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))
        # self.out_conv = nn.Linear(n_head*key_feature_dim, feature_dim)  # bias=False

    def forward(self, query=None, key=None, value=None):
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                concat = self.head[N](query, key, value)
                isFirst = False
            else:
                concat = torch.cat((concat, self.head[N](query, key, value)), -1)
        # output = self.out_conv(concat)
        output = concat
        return output
    

class RelationUnit(nn.Module):
    def __init__(self, feature_dim=12, key_feature_dim=64, patch_size=8, heads=1):
        super(RelationUnit, self).__init__()
        self.heads = heads
        self.dim = patch_size ** 2 * feature_dim
        self.hidden_dim = self.dim // heads
        self.patch_size = patch_size

        self.to_q = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1, groups=feature_dim)
        self.to_k = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1, groups=feature_dim)
        self.to_v = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1)

        self.conv = nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1)

        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)
        self.patch2feat = torch.nn.Fold(output_size=(96, 96), kernel_size=patch_size, padding=0, stride=patch_size)
        self.patch2feattest = torch.nn.Fold(output_size=(224, 336), kernel_size=patch_size, padding=0, stride=patch_size)
        self.temp = 30
        self.WK = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        # self.WQ = nn.Linear(feature_dim, key_feature_dim)
        self.WV = nn.Linear(feature_dim, feature_dim)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        '''
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        '''
        
        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        

    def forward(self, query=None, key=None, value=None):
        b, t_q, c, h, w = query.shape  # B, 3, 16, 64, 64
        b, t, c, h, w = key.shape
        H, D = self.heads, self.dim
        d = self.hidden_dim
        n = (h // self.patch_size) * (w // self.patch_size)

        q = self.to_q(query.view(-1, c, h, w))  # [B*3, C, 64, 64]
        k = self.to_k(key.view(-1, c, h, w))  # [B*3, C, 64, 64]
        v = self.to_v(value.view(-1, c, h, w))  # [B*3, C, 64, 64]

        unfold_q = self.feat2patch(q)  # [B*3, 8*8*C, 8*8]
        unfold_k = self.feat2patch(k)  # [B*3, 8*8*C, 8*8]
        unfold_v = self.feat2patch(v)  # [B*3, 8*8*C, 8*8]

        unfold_q = unfold_q.view(b, t_q, H, d, n)  # [B, 3, H, 8*8*C/H, 8*8]
        unfold_k = unfold_k.view(b, t, H, d, n)  # [B, 3, H, 8*8*C/H, 8*8]
        unfold_v = unfold_v.view(b, t, H, d, n)  # [B, 3, H, 8*8*C/H, 8*8]

        unfold_q = unfold_q.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*16/H, 3, 8*8]
        unfold_k = unfold_k.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*16/H, 3, 8*8]
        unfold_v = unfold_v.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 8*8*16/H, 3, 8*8]

        unfold_q = unfold_q.view(b, H, d, t_q * n)  # [B, H, 8*8*16/H, Tq*8*8]
        unfold_k = unfold_k.view(b, H, d, t * n)  # [B, H, 8*8*16/H, 3*8*8]
        unfold_v = unfold_v.view(b, H, d, t * n)  # [B, H, 8*8*16/H, 3*8*8]

        attn = torch.matmul(unfold_q.transpose(2, 3), unfold_k)  # [B, H, Tq*8*8, 3*8*8]
        attn = attn * (d ** (-0.5))  # [B, H, 3*8*8, 3*8*8]
        attn = F.softmax(attn*self.temp, dim=-1)  # [B, H, 3*8*8, 3*8*8]

        attn_x = torch.matmul(attn, unfold_v.transpose(2, 3))  # [B, H, Tq*8*8, 8*8*C/H]
        attn_x = attn_x.view(b, H, t_q, n, d)  # [B, H, Tq, 8*8, 8*8*64/H]
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()  # [B, Tq, H, 8*8*C/H, 8*8]
        attn_x = attn_x.view(b * t_q, D, n)  # [B*Tq, 8*8*C, 8*8]
        if h < 100:
            feat = self.patch2feat(attn_x)  # [B*Tq, C, 64, 64]
        else:
            feat = self.patch2feattest(attn_x)  # [B*Tq, C, 64, 64]

        #output = self.conv(feat).view(query.shape)  # [B, Tq, C, 64, 64]
        output = feat.view(query.shape)  # [B, Tq, C, 64, 64]


        return output
