# Author: Muhammad Anjum

import os
import cv2 as cv

SIGN_IMGS_DIR = "sign_imgs/"

SIGN_IMG_H = 128
SIGN_IMG_W = 128
SIGN_IMG_PAD = 2
SHOW_SIGN_FOR = 10

signs_to_show = {}

def load_imgs():
    sign_imgs = {}
    for img in os.listdir(SIGN_IMGS_DIR):
        sign_id = int(img.split(".")[0])
        cv_img = cv.imread(os.path.join(SIGN_IMGS_DIR, img))
        sign_imgs[sign_id] = cv_img
    
    return sign_imgs

def add_sign_img_to_img(img, sign_img, x, y):
    h, w, _ = sign_img.shape
    img[y: y + h, x: x + w] = sign_img


def show_signs(img, sign_imgs, img_w, img_h):
    global signs_to_show 

    signs_added = 0

    for sign in signs_to_show:
        if signs_to_show[sign] > 0:
            x = img_w - (signs_added // 5 + 1) * (SIGN_IMG_W + SIGN_IMG_PAD)
            y = img_h - (signs_added % 5 + 1) * (SIGN_IMG_H + SIGN_IMG_PAD)
            sign_img = sign_imgs.get(sign)
            if sign_img is not None:
                add_sign_img_to_img(img, sign_img, x, y)
                signs_added += 1
            signs_to_show[sign] -= 1
    
def add_signs_to_img(predictions, threshold=0.6):
    global sings_to_show

    for prediction in predictions:
        _, _, _, _, sign_idx, acc = prediction
        if acc > threshold:
            signs_to_show[sign_idx] = SHOW_SIGN_FOR
    
