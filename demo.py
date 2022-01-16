import time
import queue
import threading
import cv2 as cv

import model

def webcam_demo():
    model_dict = model.setup_model()
    model_params = model.default_params()

    # webcam
    webcam = cv.VideoCapture(0)

    start_time = time.time()
    frame_count = 0

    while True:
        _, img = webcam.read()
        img, _ = model.run_algorithm_on_img(img, model_dict, model_params)
        frame_count += 1

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print("Fps: ", str(round(fps, 2)))

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

def video_demo(video_path):
    model_dict = model.setup_model()
    model_params = model.default_params()

    # video
    video = cv.VideoCapture(video_path)

    fp_prediction_time = open("output/prediction_time.txt", "w")
    fp_predictions = open("output/predictions.txt", "w")

    exe_start = time.time()
    frame_count = 0

    while True:
        _, img = video.read()
        frame_count += 1

        start_time = time.process_time_ns()
        img, predictions = model.run_algorithm_on_img(img, model_dict, model_params)
        prediction_time = time.process_time_ns() - start_time

        fp_prediction_time.write(f"{prediction_time}\n")

        elapsed_time = time.time() - exe_start
        fps = frame_count / elapsed_time
        print(f"FPS: {round(fps, 2)}")

        # write prediction image and text
        if len(predictions):
            cv.imwrite(f"output/{frame_count}.jpg", img)
            fp_predictions.write(f"{frame_count}: {predictions}\n")

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
    fp_prediction_time.close()


if __name__ == "__main__":
    video_demo("input/munich.mp4")