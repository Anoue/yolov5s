#==============common.py=================
class Split_1(nn.Module):  #需要做后续处理的
    def __init__(self):
        super().__init__()
        # self.untouched_dim = c1 - self.dim
        
    def forward(self, x):
        b, c, h, w = x.size()
        return x[:, :c // 4, :, :]
    
class Split_2(nn.Module): #不需要做后续处理
    def __init__(self):
        super().__init__()
        # self.untouched_dim = c1 - self.dim
        
    def forward(self, x):
        b, c, h, w = x.size()
        return x[:, c // 4:, :, :]
#===============yolo.py===================
elif m is Split_1:           
    c2 = ch[f] // 4      
elif m is Split_2:
    c2 = ch[f] - ch[f] // 4  

#==============yaml===1/4=================
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
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
#   [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
#    [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
#    [-1, 3, C3, [128]],
#    [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
#    [-1, 6, C3, [256]],
#    [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
#    [-1, 9, C3, [512]],
#    [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
#    [-1, 3, C3, [1024]],
#    [-1, 1, SPPF, [1024, 5]],  # 9
#   ]

# # YOLOv5 v6.0 head
# head:
#   [[-1, 1, Conv, [512, 1, 1]], # 10 256 20 20 

#   [10, 1, Split_1, []], # 11 64 20 20
#   [10, 1, Split_2, []], # 12 192 20 20

#   [6, 1, Split_1, []], # 13 64 40 40
#   [6, 1, Split_2, []], # 14 192 40 40

#   [4, 1, Split_1, []], # 15 32 80 80
#   [4, 1, Split_2, []], # 16 96 80 80

#   [11, 1, nn.Upsample, [None, 2, 'nearest']], # 17 64 40 40
#   [[-1, 13], 1, Concat, [1]], # 18 128 40 40
#   [-1, 3, C3, [128, False]], # 19 64 40 40

#   [-1, 1, Conv, [64, 1, 1]], # 20 32 40 40
#   [-1, 1, nn.Upsample, [None, 2, 'nearest']], #21 32 80 80
#   [[-1, 15], 1, Concat, [1]], # 22 64 80 80
#   [-1, 3, C3, [64, False]], # 23 32 80 80

#   [[-1, 16], 1, Concat, [1]], # 24 128 80 80
#   [-1, 3, C3, [256, False]], # 25 128 80 80

#   [23, 1, Conv, [64, 3, 2]], # 26 32 40 40
#   [[-1, 20], 1, Concat, [1]], # 27 64 40 40
#   [-1, 3, C3, [128, False]], # 28 64 40 40

#   [[-1, 14], 1, Concat, [1]], # 29 256 40 40
#   [-1, 3, C3, [512, False]], # 30 256 40 40

#   [28, 1, Conv, [128, 3, 2]], # 31 64 20 20
#   [[-1, 11], 1, Concat, [1]], # 32 128 20 20
#   [-1, 3, C3, [128, False]], # 33 64 20 20
  
#   [[-1, 12], 1, Concat, [1]], # 34 256 20 20
#   [-1, 3, C3, [1024, False]], # 35 512 20 20

#   [[25, 30, 35], 1, Detect, [nc, anchors]],
#   ]
#=====================ymal===1/8=============
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
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
#   [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
#    [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
#    [-1, 3, C3, [128]],
#    [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
#    [-1, 6, C3, [256]],
#    [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
#    [-1, 9, C3, [512]],
#    [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
#    [-1, 3, C3, [1024]],
#    [-1, 1, SPPF, [1024, 5]],  # 9
#   ]

# # YOLOv5 v6.0 head
# head:
#   [[-1, 1, Conv, [512, 1, 1]], # 10 256 20 20 

#   [10, 1, Split_1, []], # 11 64 20 20
#   [10, 1, Split_2, []], # 12 192 20 20

#   [6, 1, Split_1, []], # 13 64 40 40
#   [6, 1, Split_2, []], # 14 192 40 40

#   [4, 1, Split_1, []], # 15 32 80 80
#   [4, 1, Split_2, []], # 16 96 80 80

#   [11, 1, nn.Upsample, [None, 2, 'nearest']], # 17 64 40 40
#   [[-1, 13], 1, Concat, [1]], # 18 128 40 40
#   [-1, 3, C3, [64, False]], # 19 64 40 40

#   [-1, 1, Conv, [32, 1, 1]], # 20 32 40 40
#   [-1, 1, nn.Upsample, [None, 2, 'nearest']], #21 32 80 80
#   [[-1, 15], 1, Concat, [1]], # 22 64 80 80
#   [-1, 3, C3, [32, False]], # 23 32 80 80

#   [[-1, 16], 1, Concat, [1]], # 24 128 80 80
#   [-1, 3, C3, [256, False]], # 25 128 80 80

#   [23, 1, Conv, [32, 3, 2]], # 26 32 40 40
#   [[-1, 20], 1, Concat, [1]], # 27 64 40 40
#   [-1, 3, C3, [64, False]], # 28 64 40 40

#   [[-1, 14], 1, Concat, [1]], # 29 256 40 40
#   [-1, 3, C3, [512, False]], # 30 256 40 40

#   [28, 1, Conv, [64, 3, 2]], # 31 64 20 20
#   [[-1, 11], 1, Concat, [1]], # 32 128 20 20
#   [-1, 3, C3, [64, False]], # 33 64 20 20
  
#   [[-1, 12], 1, Concat, [1]], # 34 256 20 20
#   [-1, 3, C3, [1024, False]], # 35 512 20 20

#   [[25, 30, 35], 1, Detect, [nc, anchors]],
#   ]