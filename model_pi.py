import threading
import queue
import numpy as np
import cv2 as cv
from tflite_runtime.interpreter import Interpreter

YOLO_WEIGHTS = "weights/yolov3_training_last.weights"
YOLO_CFG = "cfg/yolov3_training.cfg"
YOLO_CLASSES = "cfg/signs.names.txt"

CLASSIFIER = "weights/traffic.tflite"
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
    global CLASSIFIER_H
    global CLASSIFIER_W

    classifier = Interpreter(CLASSIFIER, num_threads=3)
    classifier.allocate_tensors()

    _, CLASSIFIER_H, CLASSIFIER_W, _ = classifier.get_input_details()[0]["shape"]
    classifier_idx = classifier.get_input_details()[0]["index"]

    print(classifier.get_input_details()[0])
    return classifier, classifier_idx

def get_classifier_output_tensors(classifier):
    output_details = classifier.get_output_details()
    tensor_list = []

    for output in output_details:
        tensor = np.squeeze(classifier.get_tensor(output["index"]))
        tensor_list.append(tensor)
    
    return tensor_list

def preprocess_img(img):
    img = img.astype(np.float32)
    return img

def run_classifier(classifier, classifier_idx, img):
    img = preprocess_img(img)
    classifier.set_tensor(classifier_idx, img)
    classifier.invoke()

    return get_classifier_output_tensors(classifier)


def get_classifier_classes():
    classifier_classes = []
    with open(CLASSIFIER_CLASSES, "r") as f:
        classifier_classes = [line.strip() for line in f.readlines()]
    return classifier_classes

def setup_model():
    yolo_net = load_yolo_net()
    yolo_classes = get_yolo_classes()
    yolo_output_layers = get_yolo_output_layers(yolo_net)

    classifier, classifier_idx = load_classifier()
    classifier_classes = get_classifier_classes()

    return {
        "yolo_net": yolo_net,
        "yolo_classes": yolo_classes,
        "yolo_output_layers": yolo_output_layers,
        "classifier": classifier,
        "classifier_idx": classifier_idx,
        "classifier_classes": classifier_classes,
    }

def default_params():
    return {
        "threshold": 0.5,
        "show_bbox": True,
        "show_label": True,
        "run_worker": True,
    }

def run_algorithm_on_img(img, model_dict, model_params, font=cv.FONT_HERSHEY_SIMPLEX):
    predictions = []
    img_height, img_width, _ = img.shape

    detections, confidences = get_yolo_detections(img, model_dict["yolo_net"], model_dict["yolo_output_layers"], threshold=model_params["threshold"])
    detection_bboxes = [get_yolo_detection_bbox(detection, img_width, img_height) for detection in detections]
    non_overlapping_idxs = cv.dnn.NMSBoxes(detection_bboxes, confidences, 0.5, 0.4)

    for idx in non_overlapping_idxs:
        x, y, w, h = detection_bboxes[idx]
        x_w = x + w
        y_h = y + h

        # limits check
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        x_w = (img_width - 1) if x_w >= img_width else x_w
        y_h = (img_height - 1) if y_h >= img_height else y_h

        if model_params["show_bbox"]:
            cv.rectangle(img, (x, y), (x_w, y_h), (255, 0, 0), 2)

        if w > 0 and h > 0:
            img_crop = img[y: y_h, x: x_w]
            img_crop = cv.resize(img_crop, (CLASSIFIER_W, CLASSIFIER_H))
            img_crop = img_crop.reshape(-1, CLASSIFIER_W, CLASSIFIER_H, 3)

            #prediction = run_classifier()model_dict["classifier"].predict(img_crop)[0]
            prediction = run_classifier(model_dict["classifier"], model_dict["classifier_idx"], img_crop)[0]
            prediction_idx = np.argmax(prediction)
            prediction_acc = prediction[prediction_idx]

            if model_params["show_label"]:
                label = f"{model_dict['classifier_classes'][prediction_idx]}: {round(prediction_acc * 100, 1)}%"
                cv.putText(img, label, (x, y), font, 0.5, (255, 0, 0), 2)

            predictions.append([x, y, x_w, y_h, prediction_idx, prediction_acc])
    return img, predictions

def algorithm_worker(inp_queue, out_queue, model_dict, model_params):
    while model_params["run_worker"]:
        try:
            inp_img = inp_queue.get(timeout=5)
            print(inp_img)
        except queue.Empty:
            continue

        out_img, predictions = run_algorithm_on_img(inp_img, model_dict, model_params)
        out_queue.put({"img": out_img, "predictions": predictions})

def setup_algorithm_thread(model_params, model_dict=setup_model()):
    inp_queue = queue.Queue()
    out_queue = queue.Queue()
    thread =  threading.Thread(target=run_algorithm_on_img, args=(inp_queue, out_queue, model_dict, model_params))

    return inp_queue, out_queue, thread