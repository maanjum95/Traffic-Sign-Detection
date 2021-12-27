import os
import numpy as np
import cv2 as cv
import tensorflow.keras.models 

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

def get_yolo_detections(img, yolo_net, yolo_output_layers, threshold=0.5):
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
    return tensorflow.keras.models.load_model(CLASSIFIER)

def get_classifier_classes():
    classifier_classes = []
    with open(CLASSIFIER_CLASSES, "r") as f:
        classifier_classes = [line.strip() for line in f.readlines()]
    return classifier_classes

def run_algorithm_on_img(img, yolo_net, yolo_output_layers, classifier, classifier_classes, threshold=0.5, font=cv.FONT_HERSHEY_SIMPLEX, show_bbox=True, show_label=True):
    prediction_class_idxs = []
    img_height, img_width, _ = img.shape

    detections, confidences = get_yolo_detections(img, yolo_net, yolo_output_layers, threshold=threshold)
    detection_bboxes = [get_yolo_detection_bbox(detection, img_width, img_height) for detection in detections]
    non_overlapping_indexes = cv.dnn.NMSBoxes(detection_bboxes, confidences, 0.5, 0.4)

    for idx in non_overlapping_indexes:
        x, y, w, h = detection_bboxes[idx]
        x_w = x + w
        y_h = y + h

        # limits check
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        x_w = (img_width - 1) if x_w >= img_width else x_w
        y_h = (img_height - 1) if y_h >= img_height else y_h

        if show_bbox:
            cv.rectangle(img, (x, y), (x_w, y_h), (255, 0, 0), 2)

        if w > 0 and h > 0:
            print(x, y, w, h, x_w, y_h, img_width, img_height)
            img_crop = img[y: y_h, x: x_w]
            img_crop = cv.resize(img_crop, (CLASSIFIER_W, CLASSIFIER_H))
            img_crop = img_crop.reshape(-1, CLASSIFIER_W, CLASSIFIER_H, 3)

            prediction = np.argmax(classifier.predict(img_crop))
            prediction_class_idxs.append(prediction)

            if show_label:
                label = str(classifier_classes[prediction])
                cv.putText(img, label, (x, y), font, 0.5, (255, 0, 0), 2)
    return img, prediction_class_idxs


