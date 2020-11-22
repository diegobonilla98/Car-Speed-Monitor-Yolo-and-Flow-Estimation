import sys
sys.path.append('../../../PycharmProjects/Yolov3')
from models import *
from utils.utils import *
from utils.datasets import *
import cv2
import torch
from torch.autograd import Variable
import time
import numpy as np


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
image_size = 320

model = Darknet("../../../PycharmProjects/Yolov3/config/yolov3.cfg", img_size=image_size).to(device)
model.load_darknet_weights("/media/bonilla/HDD_2TB_basura/models/Yolov3_pytorch/yolov3.weights")

model.eval()
classes = load_classes("../../../PycharmProjects/Yolov3/data/coco.names")
Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor

colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
a = []


def detect_cars(frame):
    RGBimg = frame[:, :, ::-1].copy()
    imgTensor = transforms.ToTensor()(RGBimg)
    imgTensor, _ = pad_to_square(imgTensor, 0)
    imgTensor = resize(imgTensor, image_size)
    imgTensor = imgTensor.unsqueeze(0)
    imgTensor = Variable(imgTensor.type(Tensor))

    with torch.no_grad():
        detections = model(imgTensor)
        detections = non_max_suppression(detections, 0.8, 0.4)

    for detection in detections:
        if detection is not None:
            detection = rescale_boxes(detection, image_size, RGBimg.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                yield torch.stack([x1, y1, x2, y2, cls_pred]).cpu().detach().numpy().astype(np.int32)

