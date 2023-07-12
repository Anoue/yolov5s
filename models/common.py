# YOLOv5 ?? by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class GhostFpnBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConv(c1, c_, 1, 1)
        self.cv2 = DWConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
# class GhostFpnCSP(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, autopad(1, 1), groups=g, bias=False)
#         #self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.SiLU()
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
    
class C3GhostFpn(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(GhostPBottleneck(c_, c_) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

        b, n, h, w = x.data.size()
        b_n = b * n // 2
        y = x.reshape(b_n, 2, h*w)
        y = y.permute(1, 0 ,2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)
        return y
    
        # return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.se = SeBlock(c2)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)





class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

class GPconv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1 ,g=1, act=True):
        super().__init__()
        self.c_ = c1 // 2
        self.cv1 = Conv(self.c_, self.c_ // 2, k, s, None, g, act)
        self.cv2 = Conv(self.c_, self.c_ // 2, k, s, None, self.c_ // 2 ,act)
        self.cv3 = Conv(c1, c2, 1, 1)

    def forward(self, x):
        x1, x2 = torch.split(x,[self.c_ , self.c_], dim=1)
        x1 = self.cv1(x1)
        y = torch.cat((x1, self.cv2(x1), x2), 1)
        return self.cv3(y)
        
class Ghost_shuffle_Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.dim_conv3 = c1 // 4
        self.dim_untouched = c1 - self.dim_conv3

        # self.cv1 = nn.Sequential(nn.Conv2d(self.dim_conv3, self.dim_conv3, k, s, autopad(k, p), groups=g, bias=False),
        #                          nn.BatchNorm2d(self.dim_conv3),
        #                          nn.ReLU(inplace=True))
        # self.cv2 = nn.Sequential(nn.Conv2d(self.dim_untouched, self.dim_untouched, k, s, autopad(k, p), groups=self.dim_untouched, bias=False),
        #                          nn.BatchNorm2d(self.dim_untouched),
        #                          nn.ReLU(inplace=True))
        # self.cv3 = nn.Sequential(nn.Conv2d(c1, c2, k, s, autopad(k, p), g, bias=False),
        #                          nn.BatchNorm2d(c2),
        #                          nn.ReLU(inplace=True))
        self.cv1 = Conv(self.dim_conv3, self.dim_conv3, True)
        self.cv2 = DWConv(self.dim_untouched, self.dim_untouched, True)
        self.cv3 = Conv(c1, c2, 1, 1, False)
    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.cv3(x)

        b, n, h, w = x.data.size()
        b_n = b * n // 2
        y = x.reshape(b_n, 2, h*w)
        y = y.permute(1, 0 ,2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)

        return y
    
class GhostBottleneck(nn.Module):
    
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        
        #self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
        #                              Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        self.shortcut = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
       # print(self.conv(x).shape, self.shortcut(x).shape)
        return self.conv(x) + self.shortcut(x)

class ghostbottleneck(nn.Module):
    
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear

    def forward(self, x):
        return self.conv(x) 
    
class ghost_shuffle_bottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(Ghost_shuffle_Conv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  Ghost_shuffle_Conv(c_, c2, 1, 1, act=False))  # pw-linear

    def forward(self, x):
        return self.conv(x) 
    
class GhostBottleneckStack(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.blocks = nn.ModuleList()
        self.blocks.append(ghostbottleneck(c_, c_, k ,s))
        
        if n > 1:
            for _ in range(n - 1):
                self.blocks.append(ghostbottleneck(c_, c_, k, s))
        
    def forward(self, x):
        redusial = x
        sums = []
        x = self.cv1(x)
        sums.append(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                sums.append(x)    
        for sum_ in reversed(sums):
            x += sum_
        x2 = self.cv2(redusial)
        x = torch.cat((x, x2), 1)
        x = self.cv3(x)
        return x

#class GhostBottleneckELANStack(nn.Module):
#    def __init__(self, c1, c2, n=1, k=3, s=1):
#        super().__init__()

#        c_ = c2 // 4
#        c__ = c2 // (2 * n)
#        self.n = n
#        self.cv1 = Conv(c1, c_, 1, 1)
#        self.cv2 = Conv(c1, c_, 1, 1, act=False)
#        self.cv3 = Conv(4 * c_, c2, 1)
#        self.m1 = GhostBottleneck(c_, c__)
        #self.se = SeBlock(c2)
#        if n > 1:
#            self.blocks = nn.Sequential(*(GhostBottleneck(c__, c__) for _ in range(n - 1)))
        
#    def forward(self, x):
#        redusial = x
#        sums = []
#        x = self.cv1(x)
#        sums.append(x)
#        x1 = self.m1(x)
#        sums.append(x1)
#        if self.n > 1:
#            for  block in self.blocks:
#                x1 = block(x1)
#                sums.append(x1)
#        x2 = self.cv2(redusial)
#        sums.append(x2)
        #out = self.se(torch.cat(sums, 1))#
#        out = torch.cat(sums, 1)
#        return out

class GhostBottleneckELANStack(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, s=1):
        super().__init__()

        c_ = c2 // 4
        c__ = 2 * c2 // (2 * n)
        self.n = n
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = GhostBottleneck(c_, c__)
        self.se = SeBlock(c2)
        if n > 1:
            self.blocks = nn.Sequential(*(GhostBottleneck(c__, c__) for _ in range(n - 1)))
        
    def forward(self, x):
        redusial = x
        sums = []
        x = self.cv1(x)
        sums.append(x)
        x1 = self.m1(x)
        sums.append(x1)
        if self.n > 1:
            for i, block in enumerate(self.blocks):
                x1 = block(x1)
                if i % 2 != 0:
                    sums.append(x1)
        x2 = self.cv2(redusial)
        sums.append(x2)
        out = self.se(self.cv3(torch.cat(sums, 1)))
        return out

class GhostBottleneckELANStack1(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        c_ = c2 // 4

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = GhostBottleneck(c_, 2* c_)
        self.se = SeBlock(c2)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        return self.se(self.cv3(torch.cat((x1, x2, x3), 1)))

class GhostBottleneckELANStack2(nn.Module):
    def __init__(self, c1, c2,k=3, s=1):
        super().__init__()

        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = GhostBottleneck(c_, c_)
        self.se = SeBlock(c2)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        x4 = self.m1(x3)
        return self.se(self.cv3(torch.cat((x1, x2, x3, x4), 1)))
    
class GhostBottleneckELANStack3(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        c_ = c2 // 4

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = GhostBottleneck(c_, c_)
        self.m2 = GhostBottleneck(c_ , c_ // 2)
        self.m3 = GhostBottleneck(c_ // 2, c_ // 2)
        self.se = SeBlock(c2)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        x4 = self.m2(x3)
        x5 = self.m3(x4)
        return self.se(self.cv3(torch.cat((x1, x2, x3, x4, x5), 1)))
class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale*k_up)**2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
                                padding=k_up//2*scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)                         # b * 25 * h_ * w_

        X = self.upsmp(X)                               # b * c * h_ * w_
        X = self.unfold(X)                              # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])    # b * c * h_ * w_
        return X

class GhostConv_add_concat(nn.Module):
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, p, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y) + y], 1) 
    
class GAhostBottleneck(nn.Module):
    
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv_add_concat(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv_add_concat(c_, c2, 1, 1, act=False))  # pw-linear
        
        #self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
        #                              Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        self.shortcut = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
       # print(self.conv(x).shape, self.shortcut(x).shape)
        return self.conv(x) + self.shortcut(x)
    
class GAhostBottleneckELANStack1(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        c_ = c2 // 4

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = GAhostBottleneck(c_, 2* c_)
        self.se = SeBlock(c2)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        return self.se(self.cv3(torch.cat((x1, x2, x3), 1)))
class GAhostBottleneckELANStackn(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, s=1):
        super().__init__()

        c_ = c2 // 4
        self.n = n
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv((2*n+2)*c_, c2, 1)

        self.m = nn.ModuleList()
        self.m1 = GAhostBottleneck(c_, 2*c_)
        if n > 1:
            for _ in range(n - 1):
                self.m.append(GAhostBottleneck(2*c_, 2*c_))
        # self.se = SeBlock(c2)
        # self.ema = EMA(c2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        output = x3
        if self.n > 1:
            for m_ in self.m:
                x4 = m_(x3)
                output = torch.cat((output, x4), 1)
                x3 = x4
            x3 = output
        #return self.se(self.cv3(torch.cat((x1, x2, x3), 1)))
        # return self.ema(self.cv3(torch.cat((x1, x2, x3), 1)))
        return self.cv3(torch.cat((x1, x2, x3), 1))

class GAhostBottleneckELANStack1_ema(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        c_ = c2 // 4

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = GAhostBottleneck(c_, 2* c_)
        self.ema = EMA(c2)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        return self.ema(self.cv3(torch.cat((x1, x2, x3), 1)))
    
class GAhostBottleneckELANStack2(nn.Module):
    def __init__(self, c1, c2,k=3, s=1):
        super().__init__()

        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = GAhostBottleneck(c_, c_)
        self.se = SeBlock(c2)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        x4 = self.m1(x3)
        return self.se(self.cv3(torch.cat((x1, x2, x3, x4), 1)))
    
class GAhostBottleneckELANStack3(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        c_ = c2 // 4

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = GAhostBottleneck(c_, c_)
        self.m2 = GAhostBottleneck(c_ , c_ // 2)
        self.m3 = GAhostBottleneck(c_ // 2, c_ // 2)
        self.se = SeBlock(c2)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        x4 = self.m2(x3)
        x5 = self.m3(x4)
        return self.se(self.cv3(torch.cat((x1, x2, x3, x4, x5), 1)))
class GhostELANFPNv2(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        c_ = c2 // 4
        c__ = c_ // 2
        self.cv1 = Conv(c1, c_, k, s)
        self.cv2 = DWConv(c1, c_, k, s)
        self.cv3 = Conv(c_, c_, k, s)
        self.cv4 = GhostConv_add_concat(c_, c_, 1, 1)
        self.cv5 = GhostConv_add_concat(c2, c2, 1, 1)
        #self.se = SeBlock(c2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x1)
        x4 = self.cv4(x3)
        x = torch.cat([x1, x2 + x1, x3, x4], 1)
        return self.cv5(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )        
    def forward(self, x):
        x = self.upsample(x)
        return x
    
class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, 2, 2, 0)
        )
    
    def forward(self, x):
        x = self.downsample(x)
        return x
    
class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, 4, 4, 0)
        )
    
    def forward(self, x):
        x = self.downsample(x)
        return x

class ASFF_2(nn.Module):
    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = [
            c1_l,
            c1_h
        ]
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            # self.compress_level_0 = Conv(c1_l, self.inter_dim, 1)
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
        if level == 1:
            # self.upsample = Upsample(c1_l, self.inter_dim)
            # # self.compress_level_0 = Conv(c1_h, self.inter_dim, 1)
            self.stride_level_0 = Downsample_x2(c1_l, self.inter_dim)
        
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1


        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weights_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        # print(x_level_0.shape, levels_weight[:, 0:1, :, :].shape)
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :]
        out = self.conv(fused_out_reduced)

        return out
    
class ASFF_3(nn.Module):
    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = [
            c1_l,
            c1_m,
            c1_h
        ]
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            # self.compress_level_0 = Conv(c1_l, self.inter_dim, 1)
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)
        if level == 1:
            # self.upsample = Upsample(c1_l, self.inter_dim)
            # # self.compress_level_0 = Conv(c1_h, self.inter_dim, 1)
            self.stride_level_0 = Downsample_x2(c1_l, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)
        if level == 2:
            self.stride_level_0 = Downsample_x4(c1_l, self.inter_dim)
            self.stride_level_1 = Downsample_x2(c1_m, self.inter_dim)
        
        
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weights_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)


        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]
        
        out = self.conv(fused_out_reduced)

        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c2, momentum=0.1)
        self.act = nn.SiLU(inplace=True)
        self.cv2 = nn.Conv2d(c2, c2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2, momentum=0.1)
    
    def forward(self, x):
        residual = x

        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.cv2(x)
        x = self.bn2(x)

        x += residual
        x = self.act(x)

        return x


class GhostELANFPNEMA(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        c_ = c2 // 4
        # c__ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s)
        self.cv2 = DWConv(c1, c_, k, s)
        self.cv3 = Conv(c_, c_, k, s)
        self.cv4 = GhostConv_add_concat(c_, c_, 1, 1)
        self.cv5 = GhostConv_add_concat(c2, c2, 1, 1)
        self.ema = EMA(c2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x1)
        x4 = self.cv4(x3)
        x = torch.cat([x1, x2 + x1, x3, x4], 1)
        #shuffle

        return self.ema(self.cv5(x))
            
# class GhostBottleneckELANStack(nn.Module):
#     def __init__(self, c1, c2, n=1, k=3, s=1):
#         super().__init__()

#         c_ = c2 // 4
#         c__ = c2 // (2 * n)
#         self.add = c_ == c__
#         self.n = n
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1, act=False)
#         self.cv3 = Conv(4 * c_, c2, 1)
#         self.m1 = ghostbottleneck(c_, c__)
#         if n > 1:
#             self.blocks = nn.Sequential(*(ghostbottleneck(c__, c__) for _ in range(n - 1)))
#         self.shorcut = Conv(c_, c__, 1, 1, act=False) if self.add == False else nn.Identity()
#     def forward(self, x):
#         redusial = x
#         sums = []
#         x = self.cv1(x)
#         sums.append(x)
#         x = self.m1(x)
#         if self.n > 1:
#             for block in self.blocks:
#                 sums.append(x)
#                 x = block(x)
#         for i, sum_ in enumerate(reversed(sums)):
#             if self.add:
#                 x += sum_
#             else:
#                 if i != len(sums) - 1:
#                     x += sum_
#                 else:
#                     x += self.shorcut(sum_)
#         sums.append(x)       
#         x2 = self.cv2(redusial)
#         sums.append(x2)
#         out = torch.cat(sums, 1)
#         return out

class GhostELANFPN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        c_ = c2 // 4
        c__ = c_ // 2
        self.cv1 = Conv(c1, c_, k, s)
        self.cv2 = DWConv(c1, c_, k, s)
        self.cv3 = Conv(c_, c_, k, s)
        self.cv4 = Conv(c_, c__, 3, 1)
        self.cv5 = Conv(c__, c__, k, s, None, g=c__)
        self.se = SeBlock(c2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x1)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x = torch.cat([x1, x2, x3, x4, x5], 1)
        #shuffle
        b, n, h, w = x.data.size()
        b_n = b * n // 2
        y = x.reshape(b_n, 2, h*w)
        y = y.permute(1, 0 ,2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)

        return self.se(y)
import torch.nn.functional as F
class ASFF(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.level = level
        # self.type = type
        asff_channel = 2
        expand_kernel = 3
        multiplier = 1

        self.dim = [
            int(512 * multiplier),
            int(256 * multiplier),
            int(128 * multiplier),
        ] 

        self.inter_dim = self.dim[self.level]

        # if self.type=='ASFF':
        if level == 0:
            self.stride_level_1 = GhostConv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = GhostConv(int(128 * multiplier), self.inter_dim, 3, 2)

        elif level == 1:
            self.compress_level_0 = GhostConv(int(512 * multiplier), self.inter_dim, 1, 1)

            self.stride_level_2 = GhostConv(int(128 * multiplier), self.inter_dim, 3, 2)

        elif level == 2:
            self.compress_level_0 = GhostConv(int(512 * multiplier), self.inter_dim, 1, 1)

            self.compress_level_1 = GhostConv(int(256 * multiplier), self.inter_dim, 1, 1)

        # add expand layer
        self.expand = DWConv(self.inter_dim, self.inter_dim, expand_kernel, 1)

        self.weight_level_0 = GhostConv(self.inter_dim, asff_channel, 1, 1)
        self.weight_level_1 = GhostConv(self.inter_dim, asff_channel, 1, 1)
        self.weight_level_2 = GhostConv(self.inter_dim, asff_channel, 1, 1)

        self.weight_levels = Conv(asff_channel * 3, 3, 1, 1)

    def expand_channel(self, x):
        #print('expand_channel x is', x.shape)
        # [b,c,h,w] -> [b, c*4, h/2, w/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]

        x = torch.cat([patch_top_left, patch_bot_left, patch_top_right, patch_bot_right], dim=1)
        return x
    
    def mean_channel(self, x):
        # [b,c,h,w] -> [b, c/4, h*2, w*2]
        x1 = x[:,::2, :, :]
        x2 = x[:, 1::2, :, :]
        return (x1 + x2) / 2
    
    def forward(self, x_level_0, x_level_1, x_level_2):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        # print(x[0].shape, x[1].shape, x[2].shape, self.level)
        # x_level_0 = x[2]  # max feature level [512,20,20]
        # x_level_1 = x[1]  # mid feature level [256,40,40]
        # x_level_2 = x[0]  # min feature level [128,80,80]

        #if self.type == 'ASFF':
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(
                level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        # else:
        # if self.level == 0:
        #     level_0_resized = x_level_0
        #     level_1_resized = self.expand_channel(x_level_1)
        #     level_1_resized = self.mean_channel(level_1_resized)
        #     level_2_resized = self.expand_channel(x_level_2)
        #     level_2_resized = F.max_pool2d(
        #         level_2_resized, 3, stride=2, padding=1)
        # elif self.level == 1:
        #     level_0_resized = F.interpolate(
        #         x_level_0, scale_factor=2, mode='nearest')
        #     level_0_resized = self.mean_channel(level_0_resized)
        #     level_1_resized = x_level_1
        #     level_2_resized = self.expand_channel(x_level_2)
        #     level_2_resized = self.mean_channel(level_2_resized)

        # elif self.level == 2:
        #     level_0_resized = F.interpolate(
        #         x_level_0, scale_factor=4, mode='nearest')
        #     level_0_resized = self.mean_channel(
        #         self.mean_channel(level_0_resized))
        #     level_1_resized = F.interpolate(
        #         x_level_1, scale_factor=2, mode='nearest')
        #     level_1_resized = self.mean_channel(level_1_resized)
        #     level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:
                                                            1, :, :] + level_1_resized * levels_weight[:,
                                                                                                    1:
                                                                                                    2, :, :] + level_2_resized * levels_weight[:,
                                                                                                                                                2:, :, :]
        out = self.expand(fused_out_reduced)
        #print('out is', out.shape)
        return out

class Ghost_shuffle_BottleneckStack(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, s=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ghost_shuffle_bottleneck(c1, c2, k ,s))
        
        if n > 1:
            for _ in range(n - 1):
                self.blocks.append(ghost_shuffle_bottleneck(c2, c2, k, s))
        # print(n, self.blocks)
        self.shortcut = DWConv(c2, c2, 1, 1)

    def forward(self, x):
        # out = 0
        sums = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                sums.append(self.shortcut(x))    

        for sum in sums:
            x += sum
        return x
    
class C3Ghost(C3):
# C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class C3Ghost_GAConv(C3):
# C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GAhostBottleneck(c_, c_) for _ in range(n)))

#======================================================
class GhostELAN(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 4
        c__ = c_ // 2
        self.cv1 = Conv(c1, c_, 1, s, None, g, act)
        self.cv2 = Conv(c1, c_, 1, s, None, g, act)
        self.cv3 = GhostConv(c_, c__, k, s)
        self.cv4 = GhostConv(c__, c__, k, s)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv4(x4)
        x6 = self.cv4(x5)
        x = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        #shuffle
        b, n, h, w = x.data.size()
        b_n = b * n // 2
        y = x.reshape(b_n, 2, h*w)
        y = y.permute(1, 0 ,2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)
        return y

class ghost_dwconv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, groups=math.gcd(c1, c2), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
        

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GhostPModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.oup = c2
        self.s = s
        init_channels = c2 // 2
        #new_channels = init_channels*(ratio-1)
        self.dim_conv = init_channels // 4
        self.dim_untouched = init_channels - self.dim_conv
        #self.cv1 = Conv(c1, c_, k, s, act=act)
        if act:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(c1, init_channels, k, s, autopad(s), bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True)
            )
            self.cheap_conv = nn.Sequential(
                nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, autopad(3), groups=self.dim_conv, bias=False),
                nn.BatchNorm2d(self.dim_conv),
                nn.ReLU(inplace=True)
            )
        else:
            self.primary_conv = nn.Conv2d(c1, init_channels, k, s, autopad(s), bias=False)
            self.cheap_conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, autopad(3), groups=self.dim_conv, bias=False)
        # self.primary_conv = Conv(c1, init_channels, k, s, None, g=1, act=act)
        # self.cheap_conv = Conv(self.dim_conv, self.dim_conv, 3, 1, None, g=self.dim_conv, act=act)
        # if self.s == 2:
        #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.primary_conv(x)
        y1, y2 = torch.split(x1, [self.dim_conv, self.dim_untouched], dim=1)
        y1 = self.cheap_conv(y1)
        y = torch.cat([x1, y1, y2], dim=1)
        # if self.s == 1:
        #     y = torch.cat([x1, y], dim=1)
        # elif self.s == 2:
        #     y = torch.cat([x1, self.maxpool(y)], 1)

        #shuffle
        # b, n, w, h = y.data.size()
        # b_n = b * n // 2
        # out = y.reshape(b_n, 2, h*w)
        # out = out.permute(1, 0 ,2)
        # out = out.reshape(2, -1, n // 2, h, w)
        # out = torch.cat((out[0], out[1]), 1)
        # return out
        return y
class GhostPBottleneck(nn.Module):
    
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostPModule(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostPModule(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        
        return self.conv(x) + self.shortcut(x)
class C3GhostP(C3):
# C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostPBottleneck(c_, c_) for _ in range(n)))
    
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import List

class Partial_conv3(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, k, s, bias=False)
        # self.cv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2))
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        # x = self.cv1(torch.cat((x1, x2), 1))
        x = torch.cat((c1, c2), 1)
        return x

# class EMA(nn.Module):
#     def __init__(self, channels, factor=8):
#         super(EMA, self).__init__()
#         self.groups = factor
#         assert channels // self.groups > 0
#         self.softmax = nn.Softmax(-1)
#         # self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
#         self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         b, c, h, w = x.size()
#         if torch.is_tensor(h):
#             h = h.item()
#             w = w.item()
#         group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
#         group_x_h, group_x_w = group_x.size()[2], group_x.size()[3]
#         if torch.is_tensor(group_x_h):
#             group_x_h, group_x_w = group_x_h.item(), group_x_w.item()
#         x_h = F.avg_pool2d(group_x, kernel_size=(1, group_x_w))
#         x_w = F.avg_pool2d(group_x, kernel_size=(group_x_h, 1)).permute(0, 1, 3, 2)
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
#         x_h, x_w = torch.split(hw, [h, w], dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv3x3(group_x)
#         x1_h, x1_w = x1.size()[2], x1.size()[3]
#         if torch.is_tensor(x1_h):
#             x1_h, x1_w = x1_h.item(), x1_w.item()
        
#         x11 = self.softmax(F.avg_pool2d(x1, kernel_size=(x1_h, x1_w)).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         x2_h, x2_w = x2.size()[2], x2.size()[3]
#         if torch.is_tensor(x2_h):
#             x2_h, x2_w = x2_h.item(), x2_w.item()
#         x21 = self.softmax(F.avg_pool2d(x2, kernel_size=(x2_h, x2_w)).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        # self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = GhostConv_add_concat(channels // self.groups, channels // self.groups, k=1, s=1, p=0, g=1, act=False)
        self.conv3x3 = GhostConv_add_concat(channels // self.groups, channels // self.groups, k=3, s=1, p=1, g=1, act=False)

    def forward(self, x):
        b, c, h, w = x.size()
        # if torch.is_tensor(h):
        #     h = h.item()
        #     w = w.item()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # group_x_h, group_x_w = group_x.size()[2], group_x.size()[3]
        # if torch.is_tensor(group_x_h):
        #     group_x_h, group_x_w = group_x_h.item(), group_x_w.item()
        # x_h = F.avg_pool2d(group_x, kernel_size=(1, group_x_w))
        x_h = torch.mean(group_x, 3, keepdim=True)
        # x_w = F.avg_pool2d(group_x, kernel_size=(group_x_h, 1)).permute(0, 1, 3, 2)
        x_w = torch.mean(group_x, 2, keepdim=True).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        # x1_h, x1_w = x1.size()[2], x1.size()[3]
        # if torch.is_tensor(x1_h):
        #     x1_h, x1_w = x1_h.item(), x1_w.item()
        # x11 = self.softmax(F.avg_pool2d(x1, kernel_size=(x1_h, x1_w)).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x11 = self.softmax(torch.mean(x1, (2, 3), keepdim=True).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # x2_h, x2_w = x2.size()[2], x2.size()[3]
        # if torch.is_tensor(x2_h):
        #     x2_h, x2_w = x2_h.item(), x2_w.item()
        # x21 = self.softmax(F.avg_pool2d(x2, kernel_size=(x2_h, x2_w)).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x21 = self.softmax(torch.mean(x2, (2, 3), keepdim=True).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
class none_dense_concat(nn.Module):
    def __init__(self, c1, c2, n=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
        self.cv3 = Conv(c_, c_ // 2, 1, 1)

        self.m = nn.Sequential(*(GhostBottleneck(c_ // 2, c_ // 2) for _ in range(n)))

        self.cv4 = Conv(c_, c_ // 2, 1, 1, act=False)
    def forward(self, x):
        x = self.cv1(x)
        x1 = x.clone()
        x = self.cv2(x)
        x2 = x.clone()
        x = self.cv3(x)
        x = self.m(x)
        x2 = self.cv4(x2)        
        x = torch.cat([x1, x2, x], 1)
        return x

class head_none_dense_concat(nn.Module):
    def __init__(self, c1, c2, n=1, act=True):
        super().__init__()
        # c_ = c2
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = self.m = nn.Sequential(*(GhostBottleneck(c2, c2) for _ in range(n)))

    def forward(self, x):
        x = self.cv1(x)
        x = self.m(x)
        return x
    
class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 
                 pconv_fw_type='split_cat'
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        # shortcut = x
        x = self.spatial_mixing(x)
        x = self.drop_path(self.mlp(x))
        # x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        # shortcut = x
        x = self.spatial_mixing(x)
        # x = shortcut + self.drop_path(
        #     self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        x = self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x
    
# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         return self.act(self.conv(x))


# class DWConv(Conv):
#     # Depth-wise convolution class
#     def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)
# class GhostConv(nn.Module):
    
#     def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
#         super().__init__()
#         c_ = c2 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, k, s, None, g, act)
#         self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

#     def forward(self, x):
#         y = self.cv1(x)
#         return torch.cat([y, self.cv2(y)], 1)
    
class ghostmodule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super().__init__()
        c_ = c2 // 2 # hiden channels
        self.cv1 = Conv(c1, c_, k, s,None,g)
        
        self.cv2 = Conv(c1, c_, 5, 1,None,c_)
        
    
    def forward(self, x):
        
        y = self.cv1(x)
        
        return torch.cat([y, self.cv2(x)], 1)

class ghostbottleneck2(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        # self.s = s
        # self.shorthead = ghost_dwconv(c1, c2, 1, 1)
        # self.ghost1 = ghostmodule(c1, c_, k, s)
        # self.bn1 = nn.BatchNorm2d(c_)
        # self.relu1 = nn.ReLU(inplace=True)
        # if s == 2:
        #     self.dwconv = ghost_dwconv(c_, c_, k, s, g=math.gcd(c1, c2))
        # self.ghost2 = ghostmodule(c_, c2, k, s)
        # self.bn3 = nn.BatchNorm2d(c2)
        self.ghost = nn.Sequential(ghostmodule(c1, c_, k, s),
                                #    nn.BatchNorm2d(c_),
                                #    nn.ReLU(inplace=True),
                                    SeBlock(c_),
                                   ghost_dwconv(c_, c_, k, s, g=math.gcd(c1, c2)) if s == 2 else nn.Identity(),
                                   ghostmodule(c_, c2, k, s),
                                #    nn.BatchNorm2d(c2)
        )
        #self.shortend = ghost_dwconv(c2, c2, 1, 1)

    def forward(self, x):
        
        x = self.ghost(x)
        
        return x


class none_dense_ghostcsp(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.n = n
        c_ = c2 // 2 #hidden channels
        #cc = c_ // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        
        self.m = ghostbottleneck2(c_, c_)
        self.conv = ghost_dwconv(c_, c_, 1, 1)
        
        # self.m = nn.Sequential(*(ghostbottleneck2(c_, c_) for _ in range(n)))
        self.bn = nn.BatchNorm2d(2 * c_)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        redusial = x
        x = self.cv1(redusial)
        y = self.cv2(redusial)
        
        if self.n == 3:
            x1 = self.conv(x)
            x = self.m(x)
            x_input = x + x1
        if self.n == 6:
            x1 = self.conv(x)
            x = self.m(x)
            x2 = self.conv(x)
            x = self.m(x)
            x_input = x + x1 + x2
        if self.n == 9:
            x1 = self.conv(x)
            x = self.m(x)
            x2 = self.conv(x)
            x = self.m(x)
            x3 = self.conv(x)
            x = self.m(x)
            x_input = x + x1 + x2 + x3
        return self.relu(self.bn(self.cv3(torch.cat([x_input, y], 1))))
        
# class none_dense_ghostcsp2(nn.Module):
#     def __init__(self, c1, c2, n=1):
#         super().__init__()
#         self.n = n
#         c_ = c2 // 2 #hidden channels
#         #cc = c_ // 2
#         self.cv1 = DWConv(c1, c_, 1, 1)
#         self.cv2 = DWConv(c1, c_, 1, 1)
#         self.cv3 = DWConv(2 * c_, c2, 1, 1)
        
#         self.m = ghostbottleneck2(c_, c_)
#         self.conv = DWConv(c_, c_, 1, 1)
        
#         # self.m = nn.Sequential(*(ghostbottleneck2(c_, c_) for _ in range(n)))
#         # self.bn = nn.BatchNorm2d(2 * c_)
#         # self.relu = nn.ReLU()
    
#     def forward(self, x):
#         redusial = x
#         x = self.cv1(redusial)
#         y = self.cv2(redusial)
        
#         x1 = self.conv(x)
#         x = self.m(x)
#         x2 = self.conv(x)
#         x = self.m(x)
#         x_input = x + x1 + x2
#         return self.cv3(torch.cat([x_input, y], 1))
        
# class none_dense_ghostcsp3(nn.Module):
#     def __init__(self, c1, c2, n=1):
#         super().__init__()
#         self.n = n
#         c_ = c2 // 2 #hidden channels
#         #cc = c_ // 2
#         self.cv1 = DWConv(c1, c_, 1, 1)
#         self.cv2 = DWConv(c1, c_, 1, 1)
#         self.cv3 = DWConv(2 * c_, c2, 1, 1)
        
#         self.m = ghostbottleneck2(c_, c_)
#         self.conv = DWConv(c_, c_, 1, 1)
        
#         # self.m = nn.Sequential(*(ghostbottleneck2(c_, c_) for _ in range(n)))
#         # self.bn = nn.BatchNorm2d(2 * c_)
#         # self.relu = nn.ReLU()
    
#     def forward(self, x):
#         redusial = x
#         x = self.cv1(redusial)
#         y = self.cv2(redusial)
    
#         x1 = self.conv(x)
#         x = self.m(x)
#         x2 = self.conv(x)
#         x = self.m(x)
#         x3 = self.conv(x)
#         x = self.m(x)
#         x_input = x + x1 + x2 + x3
#         return self.cv3(torch.cat([x_input, y], 1))

class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        #���ÿ�ѧϰ���� nn.Parameter�������ǣ���һ������ѵ��������Tensorת���ɿ���ѵ��������parameter
        #���һ�������ģ��ע��ò�������Ϊ��һ���� ��model.parameters()��������Parameter
        #�Ӷ��ڲ����Ż���ʱ������Զ�һ���Ż�
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size = 1, stride = 1, padding=0)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))

class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))

class shortcut(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()    
        self.shortcut1 = DWConv(c1, c2, 3, 1, act=True)

    def forward(self, x):
        return self.shortcut1(x)

#===================GhostNet==========================
class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)
        self.Excitation = nn.Sequential()
        self.Excitation.add_module('FC1', nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1))  # 1*1�������Ч����ͬ
        self.Excitation.add_module('ReLU', nn.ReLU())
        self.Excitation.add_module('FC2', nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1))
        self.Excitation.add_module('Sigmoid', nn.Sigmoid()) #add_module ��ģ����������ģ��
    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))  #expand_as ������������ָ����״������ԭ�е�Ԫ�ؽ�������

class G_bneck(nn.Module):
    def __init__(self, c1, c2, midc, k=5, s=1, use_se = False):  # ch_in, ch_mid, ch_out, kernel, stride, use_se
        super().__init__()
        assert s in [1, 2]
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),              # Expansion
                                  DWConv(c_, c_, 3, s=2, act=False) if s == 2 else nn.Identity(),  # dw
                                  # Squeeze-and-Excite
                                  SeBlock(c_) if use_se else nn.Sequential(),
                                  GhostConv(c_, c2, 1, 1, act=False))   # Squeeze pw-linear
        self.shortcut = nn.Identity() if (c1 == c2 and s == 1) else \
                                                nn.Sequential(DWConv(c1, c1, 3, s=s, act=False), \
                                                Conv(c1, c2, 1, 1, act=False)) # ����stride=2ʱ ͨ�����ı�����
        
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
#=============PConv=====================

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        #print(x[0].shape, x[1].shape)
        return torch.cat(x, self.d)

class Split_1(nn.Module):  #需要做后续处理的
    def __init__(self):
        super().__init__()
        # self.untouched_dim = c1 - self.dim
        
    def forward(self, x):
        b, c, h, w = x.size()
        return x[:, :c // 8, :, :]
    
class Split_2(nn.Module): #不需要做后续处理
    def __init__(self):
        super().__init__()
        # self.untouched_dim = c1 - self.dim
        
    def forward(self, x):
        b, c, h, w = x.size()
        return x[:, c // 8:, :, :]
    
class Add(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        
        sum = 0
        for i in x:
            sum += i
        return sum

class Avg(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):

        return (x[0] + x[1]) / 2

class Mul(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        
        return x[0] * x[1]
    
class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False, data=None):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn
        #   OpenVINO:                       *.xml
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            import openvino.inference_engine as ie
            core = ie.IECore()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = core.read_network(model=w, weights=Path(w).with_suffix('.bin'))  # *.xml, *.bin paths
            executable_network = core.load_network(network, device_name='CPU', num_requests=1)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            if saved_model:  # SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # graph_def
                gd.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf
                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                if edgetpu:  # Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    delegate = {'Linux': 'libedgetpu.so.1',
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # Lite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
            elif tfjs:
                raise Exception('ERROR: YOLOv5 TF.js inference is not supported')
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            desc = self.ie.TensorDesc(precision='FP32', dims=im.shape, layout='NCHW')  # Tensor Description
            request = self.executable_network.requests[0]  # inference request
            request.set_blob(blob_name='images', blob=self.ie.Blob(desc, im))  # name=next(iter(request.input_blobs))
            request.infer()
            y = request.output_blobs['output'].buffer  # name=next(iter(request.output_blobs))
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
                y = y[k]  # output
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.saved_model:  # SavedModel
                y = (self.model(im, training=False) if self.keras else self.model(im)[0]).numpy()
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            else:  # Lite or Edge TPU
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.jit or self.onnx or self.engine:  # warmup types
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
