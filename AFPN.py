#==================common.py=====================================
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

class BasicBlock_n(nn.Module):
    expansion = 1
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.m = nn.Sequential(*(BasicBlock(c1, c2) for _ in range(n)))

    def forward(self, x):
        return self.m(x)
        
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
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)
        
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
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)

        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)

        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)
        
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
# #==================yaml=====================================
# # YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# # Parameters
# nc: 80  # number of classes
# depth_multiple: 0.33  # model depth multiple
# width_multiple: 0.50  # layer channel multiple
# anchors:
#   - [10,13, 16,30, 33,23]  # P3/8
#   - [30,61, 62,45, 59,119]  # P4/16
#   - [116,90, 156,198, 373,326]  # P5/32

# # YOLOv5 v6.0 backbone
# backbone:
#   # [from, number, module, args]
#   [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2  320
#    [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4   160
#    [-1, 3, C3, [128]],     #160
#    [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8  80
#    [-1, 6, C3, [256]],  #80
#    [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16  40
#    [-1, 9, C3, [512]],   #40
#    [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32  20
#    [-1, 3, C3, [1024]],  #20
#    [-1, 1, SPPF, [1024, 5]],  # 9
#   ]

# # YOLOv5 v6.0 head
# head:
#   [[4, 1, Conv, [64, 1, 1]], # 80 80
#   [6, 1, Conv, [128, 1, 1]],  # 40 40
#   [9, 1, Conv, [256, 1, 1]],  #20 20

#   [10, 1, Conv, [64, 1, 1]],  # 13 80 80
#   [11, 1, Conv, [128, 1, 1]], # 14 40 40
#   [12, 1, Conv, [256, 1, 1]], # 15 20 20
  
#   [[13, 14], 1, ASFF_2, [64, 0]],  #  16 80 80 
#   [[13, 14], 1, ASFF_2, [128, 1]],  #  17 40 40

#   [16, 3, BasicBlock_n, [64]],  # 18 80 80
#   [17, 3, BasicBlock_n, [128]], # 19 40 40

#   [[18, 19, 15], 1, ASFF_3, [64, 0]],  # 20
#   [[18, 19, 15], 1, ASFF_3, [128, 1]], # 21
#   [[18, 19, 15], 1, ASFF_3, [256, 2]], # 22

#   [20, 9, BasicBlock_n, [64]],  #23
#   [21, 9, BasicBlock_n, [128]], #24
#   [22, 9, BasicBlock_n, [256]], #25

#   [23, 1, Conv, [256, 1, 1]], #26
#   [24, 1, Conv, [512, 1, 1]], #27
#   [25, 1, Conv, [1024, 1, 1]], #28

#   [[26, 27, 28], 1, Detect, [nc, anchors]],
#   ]
#==================yolo.py=====================================
# if m in [Conv, GhostConv, BasicBlock_n]:
#     c1, c2 = ch[f], args[0]
#     if c2 != no:  # if not output
#         c2 = make_divisible(c2 * gw, 8)

#      args = [c1, c2, *args[1:]] #q: *args[1:]?   
#     if m in [BasicBlock_n]:
#         args.insert(2, n)  # number of repeats
#         n = 1
# elif m is ASFF_2:
#     c1, c2 = [ch[f[0]], ch[f[1]]], args[0]
#     if c2 != no:  # if not output
#         c2 = make_divisible(c2 * gw, 8)
#     args = [c1, c2, *args[1:]]
# elif m is ASFF_3:
#     c1, c2 = [ch[f[0]], ch[f[1]], ch[f[2]]], args[0]
#     if c2 != no:  # if not output
#         c2 = make_divisible(c2 * gw, 8)
#     args = [c1, c2, *args[1:]]
