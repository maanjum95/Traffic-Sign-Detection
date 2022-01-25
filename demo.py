import time
import queue
import threading
import cv2 as cv

import model
import signs

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
    sign_imgs = signs.load_imgs() 

    # video
    video = cv.VideoCapture(video_path)

    exe_start = time.time()
    frame_count = 0

    while True:
        _, img = video.read()
        img_h, img_w, _ = img.shape
        frame_count += 1

        img, predictions = model.run_algorithm_on_img(img, model_dict, model_params)

        # add signs imags
        signs.add_signs_to_img(predictions)
        signs.show_signs(img, sign_imgs, img_w, img_h)

        if len(predictions) > 0:
            cv.imwrite(f"output/{frame_count}.jpg", img)

        elapsed_time = time.time() - exe_start
        fps = frame_count / elapsed_time
        print(f"FPS: {round(fps, 2)}")

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            video.release()
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_demo("input/munich.mp4")