import time
import cv2 as cv

import model

def webcam_demo():
    model_dict = model.setup_model()

    # webcam
    webcam = cv.VideoCapture(0)

    start_time = time.time()
    frame_count = 0

    while True:
        _, img = webcam.read()
        img, _ = model.run_algorithm_on_img(img, model_dict)
        frame_count += 1

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print("Fps: ", str(round(fps, 2)))

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    webcam_demo()