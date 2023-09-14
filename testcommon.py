import torch.nn as nn
from models.common import *
class GhostConv_ASFF(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

        self.weights = Conv(c_, 2, 1, 1)
        self.weights_levels = nn.Conv2d(4, 2, kernel_size=1, stride=1, padding=0)
        # self.conv = Conv(c2, c2, 3, 1)

    def forward(self, x):
       
        x1 = self.cv1(x)
        x2 = self.cv2(x1)

        level_x1_weight = self.weights(x1)
        level_x2_weight = self.weights(x2)

        level_x_weight = torch.cat([level_x1_weight, level_x2_weight], 1)
        levels_weight = self.weights_levels(level_x_weight)
        levels_weight = F.softmax(levels_weight, dim=1)

        out = torch.cat([x1 * levels_weight[:, 0:1, :, :], x2 * levels_weight[:, 1:2, :, :]], 1)
        # out = self.conv(fused_out_reduced)

        return out

class GhostBottleneck_ASFF(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.add = c1 == c2
        self.conv = nn.Sequential(
            GhostConv_ASFF(c1, c_, 1, 1),  # pw
            GhostConv_ASFF(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()
        self.cv = DWConv(c1, c2, 1, 1, act=False) if self.add == False else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x) if self.add else self.conv(x) + self.cv(self.shortcut(x))


class C3Ghost_ASFF(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck_ASFF(c_, c_) for _ in range(n)))


class GhostBottleneck_CSA(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
    
class C3Ghost_CSA(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck_CSA(c_, c_) for _ in range(n)))

from einops import rearrange

class MSGConv(nn.Module):
    # Multi-Scale Ghost Conv
    def __init__(self, c1, c2, k=1, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = c2 // 2
        # assert min_ch >= 4, f'channel must Greater than {8}, but {c2}'
        self.s = s
        self.convs = nn.ModuleList([])
        self.cv1 = Conv(c1, min_ch, k, s)
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch // 2, c2=min_ch // 2, k=ks, g=min_ch // 2))
        self.conv1x1 = Conv(c2, c2, 1)

    def forward(self, x):

        x1 = self.cv1(x)
        x2 = x1
        x2 = rearrange(x2, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x2 = torch.stack([self.convs[i](x2[..., i]) for i in range(len(self.convs))])
        x2 = rearrange(x2, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1x1(x)
        return x

class MSGAConv(nn.Module):
    # Multi-Scale Ghost Conv
    def __init__(self, c1, c2, k=3, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = c2 // 2
        # assert min_ch >= 4, f'channel must Greater than {8}, but {c2}'
        self.s = s
        
        self.convs = nn.ModuleList([])
        if s==1: 
            self.cv1 = Conv(c1, min_ch, 1, 1) 
        if s==2:
            self.cv1 = Conv(c1, min_ch, 3, 2) 
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch // 2, c2=min_ch // 2, k=ks, g=min_ch // 2))
        self.conv1x1 = Conv(c2, c2, 1)
        self.add = c1 != c2
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if self.add else nn.Identity()

    def forward(self, x):

        x1 = self.cv1(x)
        x2 = x1
        x2 = rearrange(x2, 'bs (g ch) h w -> bs ch h w g', g=self.groups)

        x2 = torch.stack([self.convs[i](x2[..., i]) for i in range(len(self.convs))])
        x2 = rearrange(x2, 'g bs ch h w -> bs (g ch) h w')
        out = torch.cat([x1, x2], dim=1)
        x = self.shortcut(x)
        out = self.conv1x1(out) + x
        return out
    
class Bottleneck_MSG(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = MSGConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class MSGBottleneck(nn.Module):
    # GhostBottleneck  Conv->MSGConv
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(MSGConv(c1, c_),  # pw
                                  MSGConv(c_, c2))  # pw-linear
        self.add = c1 != c2
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if self.add else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
    
class MSGABottleneck(nn.Module):
    
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(MSGAConv(c1, c_),  # pw
                                  MSGAConv(c_, c2))  # pw-linear
        
        # self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
        #                              Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        self.add = c1 != c2
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if self.add else nn.Identity()

    def forward(self, x):

        return self.conv(x) + self.shortcut(x)
    
class C3_MSG(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_MSG(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
class C3Ghost_MSG(nn.Module):
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(MSGBottleneck(c_, c_) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
class C3GAhost_MSG(nn.Module):
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(MSGABottleneck(c_, c_) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
class MSGELAN_ema(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = MSGABottleneck(c_, 2* c_)
        self.ema = EMA(c2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        return self.ema(self.cv3(torch.cat((x1, x2, x3), 1)))
    
class MSGELAN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = MSGBottleneck(c_, 2* c_)
        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        # return self.ema(self.cv3(torch.cat((x1, x2, x3), 1)))
        
        return self.cv3(torch.cat((x1, x2, x3), 1))
    
class MSGAELAN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1, act=False)
        self.cv3 = Conv(4 * c_, c2, 1)
        self.m1 = MSGABottleneck(c_, 2* c_)
        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m1(x1)
        # return self.ema(self.cv3(torch.cat((x1, x2, x3), 1)))
        
        return self.cv3(torch.cat((x1, x2, x3), 1))
    
class EMSConv(nn.Module):
    # Efficient Multi-Scale Conv
    def __init__(self, channel=64, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'
        
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)
        
    def forward(self, x):
        _, c, _, _ = x.size()
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_group = rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        x = self.conv_1x1(x)
        
        return x

class Bottleneck_EMSC(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = EMSConv(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_EMSC(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_EMSC(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class MSGConv_Group_conv1x1(nn.Module):
    # Multi-Scale Ghost Conv
    def __init__(self, c1, c2, k=1, s=1, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = c2 // 2
        # assert min_ch >= 4, f'channel must Greater than {8}, but {c2}'
        self.s = s
        
        self.convs = nn.ModuleList([])
        self.cv1 = Conv(c1, min_ch, k, s)
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch // 2, c2=min_ch // 2, k=ks, g=min_ch // 2))
        self.conv1x1 = GSConv(c2, c2)

    def forward(self, x):

        x1 = self.cv1(x)
        x2 = x1
        x2 = rearrange(x2, 'bs (g ch) h w -> bs ch h w g', g=self.groups)

        x2 = torch.stack([self.convs[i](x2[..., i]) for i in range(len(self.convs))])
        x2 = rearrange(x2, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1x1(x)
        return x

class MSGBottleneck_Group(nn.Module):
    # GhostBottleneck  Conv->MSGConv
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(MSGConv_Group_conv1x1(c1, c_),  # pw
                                  MSGConv_Group_conv1x1(c_, c2))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
    
class C3Ghost_MSG_Group_conv1x1(nn.Module):
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(MSGBottleneck_Group(c_, c_) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
#=================ScConv CVPR2023====================
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.gamma      = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.beta       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps

    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta

class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5 
                 ):
        super().__init__()
        
        self.gn             = GroupBatchnorm2d( oup_channels, group_num = group_num )
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.gamma/sum(self.gn.gamma)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        info_mask   = reweigts>=self.gate_treshold
        noninfo_mask= reweigts<self.gate_treshold
        x_1         = info_mask * x
        x_2         = noninfo_mask * x
        x           = self.reconstruct(x_1,x_2)
        return x
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    # https://github.com/cheng-haha/ScConv/blob/main/ScConv.py
    def __init__(self,
                op_channel:int,
                group_num:int = 16,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold)
        self.CRU = CRU(op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size)
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class Bottleneck_ScConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = ScConv(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_ScConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


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


class CARAFE_MSGConv(nn.Module):
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
        super(CARAFE_MSGConv, self).__init__()
        self.scale = scale

        self.comp = MSGConv(c, c_mid)
        self.enc = MSGConv(c_mid, (scale*k_up)**2)
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
