
import os
import sys
from pathlib import Path

import cv2
import torch
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
from models.common import DetectMultiBackend

from utils.general import (LOGGER,  check_img_size, non_max_suppression,scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync
from utils.augmentations import letterbox

def detect_engine(imgs, imgsz=640, stride=64, auto=True):

    img = letterbox(imgs, imgsz, stride, auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)


    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    im, im0s = img, imgs

    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1
    pred = model(im, augment=False, visualize=False)

    t3 = time_sync()
    dt[1] += t3 - t2
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000)

    dt[2] += time_sync() - t3

    for i, det in enumerate(pred):

        seen += 1
        im0 = im0s.copy()
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):

            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label =  f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()
        
        cv2.imshow('img', im0)
        cv2.waitKey(1) 

    LOGGER.info(f'({t3 - t2:.3f}s)')

class detect():
    def __init__(self):
        rospy.init_node('yolov5s', anonymous=True)  #使用名称yolov5s初始化ROS节点
        self.bridge = CvBridge()  #初始化了CvBridge类，以将ROS图像消息转换为OpenCV可用的格式。
        # ROS订阅器，订阅/camera/color/image_raw主题，并在接收到消息时调用callback方法
        self.img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback, queue_size=1)  
    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") #接收ROS图像消息并将其转换为detect_engine函数可用的格式
        # 检测
        detect_engine(cv_image, imgsz=imgsz, stride=stride, auto=pt)

if __name__ == '__main__':
    # device = select_device('')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    engine_path = 'weights\yolov5s-ghostelan-gaconv_1_1_1.engine'
    model = DetectMultiBackend(engine_path, device=device, dnn=False, data='data/coco128.yaml')
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    half = False
    imgsz = check_img_size((640, 640), s=stride)  # check img_size
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    detect()
