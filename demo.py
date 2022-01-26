import os
import time
import queue
import threading
import cv2 as cv

if os.uname()[1] == "raspberrypi":
    import model_pi as model
    from re_terminal import setup_btns
else:
    import model
    setup_btns = None

import signs


def setup_cv_window():
    window_name = "traffig_sign_detector"
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    return window_name

def show_param_fps(img, model_params, fps):
    img_h, img_w, _ = img.shape

    # show parameters
    cv.putText(img, str(model_params), (0, img_h - 6), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # fps
    cv.putText(img, f"Fps: {round(fps, 2)}", (img_w - 60, 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def webcam_demo():
    window = setup_cv_window()
    model_dict = model.setup_model()
    model_params = model.default_params()
    sign_imgs = signs.load_imgs()

    if setup_btns is not None:
        setup_btns(model_params)

    # webcam
    webcam = cv.VideoCapture(0)

    start_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        _, img = webcam.read()
        img_h, img_w, _ = img.shape
        frame_count += 1

        # run algorithm on image
        img, predictions = model.run_algorithm_on_img(img, model_dict, model_params)

        # add signs imags
        if model_params["show_signs"]:
            signs.add_signs_to_img(predictions, model_params["sign_threshold"])
            signs.show_signs(img, sign_imgs, img_w, img_h)
        else:
            # remove signs to show
            signs.signs_to_show = {}

        # show parameters and fps
        show_param_fps(img, model_params, fps)

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print("Fps: ", str(round(fps, 2)))

        cv.imshow(window, img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    cv.destroyAllWindows()

def video_demo(video_path):
    window = setup_cv_window()
    model_dict = model.setup_model()
    model_params = model.default_params()
    sign_imgs = signs.load_imgs() 

    if setup_btns is not None:
        setup_btns(model_params)

    # video
    video = cv.VideoCapture(video_path)

    exe_start = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        _, img = video.read()
        img_h, img_w, _ = img.shape
        frame_count += 1

        # run algorithm on image
        img, predictions = model.run_algorithm_on_img(img, model_dict, model_params)

        # add signs imags
        if model_params["show_signs"]:
            signs.add_signs_to_img(predictions, model_params["sign_threshold"])
            signs.show_signs(img, sign_imgs, img_w, img_h)
        else:
            # remove signs to show
            signs.signs_to_show = {}

        # show parameters and fps
        show_param_fps(img, model_params, fps)

        #if len(predictions) > 0:
        #    cv.imwrite(f"output/{frame_count}.jpg", img)

        elapsed_time = time.time() - exe_start
        fps = frame_count / elapsed_time
        print(f"FPS: {round(fps, 2)}")

        cv.imshow(window, img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            video.release()
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_demo("input/munich.mp4")
    #webcam_demo()