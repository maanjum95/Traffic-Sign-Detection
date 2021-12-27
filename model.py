import os
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = ''

YOLO_WEIGHTS = "weights/yolov4-tiny_training_last.weights"
YOLO_CFG = "cfg/yolov4-tiny_training.cfg"
YOLO_CLASSES = "cfg/signs.names.txt"

CLASSIFIER = "weights/traffic.h5"
CLASSIFIER_CLASSES = "cfg/signs_classes.txt"
CLASSIFIER_H = 32
CLASSIFIER_W = 32

def load_yolo_net():
    return cv.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)

def get_yolo_output_layers(yolo_net):
    layers = yolo_net.getLayerNames()
    
    return [layers[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

def get_yolo_classes():
    yolo_classes = []
    with open(YOLO_CLASSES, "r") as f:
        yolo_classes = [line.strip() for line in f.readlines()]
    return yolo_classes

def get_yolo_output(yolo_net, yolo_output_layers, img):
    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo_net.setInput(blob)

    return yolo_net.forward(yolo_output_layers)

def get_yolo_detections(yolo_net, yolo_output_layers, img, threshold=0.5):
    yolo_output = get_yolo_output(yolo_net, yolo_output_layers, img)
    yolo_detections = []
    yolo_confidences = []

    for output in yolo_output:
        for detection in output:
            confidence = max(detection[5:])

            if confidence > threshold:
                yolo_detections.append(detection[:4])
                yolo_confidences.append(confidence)

    return yolo_detections, yolo_confidences

def get_yolo_detection_bbox(detection, img_width, img_height):
    center_x = int(detection[0] * img_width)
    center_y = int(detection[1] * img_height)

    w = int(detection[2] * img_width)
    h = int(detection[3] * img_height)

    x = center_x - w // 2
    y = center_y - h // 2

    return x, y, w, h


def load_classifier():
    return load_model(CLASSIFIER)

def get_classifier_classes():
    classifier_classes = []
    with open(CLASSIFIER_CLASSES, "r") as f:
        classifier_classes = [line.strip() for line in f.readlines()]
    return classifier_classes