import time
import numpy as np
import cv2 as cv

import model

def webcam_demo():
    yolo_net = model.load_yolo_net()
    yolo_classes = model.get_yolo_classes()
    yolo_output_layers = model.get_yolo_output_layers(yolo_net)

    classifier = model.load_classifier()
    classifier_classes = model.get_classifier_classes()

    # webcam
    webcam = cv.VideoCapture(0)

    font = cv.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    frame_count = 0

    while True:
        _, img = webcam.read()
        img_height, img_width, _ = img.shape
        frame_count += 1

        detections, confidences = model.get_yolo_detections(yolo_net, yolo_output_layers, img)
        detection_bboxes = []

        for detection in detections:
            detection_bboxes.append(model.get_yolo_detection_bbox(detection, img_width, img_height))

        non_overlapping_indexes = cv.dnn.NMSBoxes(detection_bboxes, confidences, 0.5, 0.4)

        for idx in non_overlapping_indexes:
            x, y, w, h = detection_bboxes[idx]
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            img_crop = img[y: y+h, x: x+w]
            if len(img_crop) > 0:
                img_crop = cv.resize(img_crop, (model.CLASSIFIER_W, model.CLASSIFIER_H))
                img_crop = img_crop.reshape(-1, model.CLASSIFIER_W, model.CLASSIFIER_H, 3)
                prediction = np.argmax(classifier.predict(img_crop))
                label = str(classifier_classes[prediction])
                cv.putText(img, label, (x, y), font, 0.5, (255, 0, 0), 2)

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print("Fps: ", str(round(fps, 2)))

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
        



if __name__ == "__main__":
    webcam_demo()