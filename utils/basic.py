'''
Basic functions
'''

import cv2
import numpy as np


def read_image_grayscale(filepath):
    img = cv2.imread(filepath)

    # turn image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    return img

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)