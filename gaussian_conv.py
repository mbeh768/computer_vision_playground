# this file demonstrates use of convolution with nth order gaussian distributions

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from utils.basic import read_image_grayscale
from utils.visualization import show_images


filepath = 'images/basic/circle_square.png'

img = read_image_grayscale(filepath)

# compute second derivative approximations via gaussian filter
sigma = 20

# visualize convolution with gaussian
I = gaussian_filter(img, sigma=sigma, order=(0, 0))

# show blurred image
show_images([img, I], ["raw_image", "blurred"], ["gray", "gray"])

# compute first derivatives
Ix = gaussian_filter(img, sigma=sigma, order=(0, 1))
Iy = gaussian_filter(img, sigma=sigma, order=(1, 0))

# show first derivatives
show_images([Ix, Iy], ["Ix", "Iy"])

# compute second derivatives
Ixx = gaussian_filter(img, sigma=sigma, order=(0, 2))
Iyy = gaussian_filter(img, sigma=sigma, order=(2, 0))
Ixy = gaussian_filter(img, sigma=sigma, order=(1, 1))

# compute eigenvalues of 2x2 hessian for each pixel
trace = Ixx + Iyy                               # trace
det = Ixx * Iyy - Ixy**2                        # determinant
disc = np.maximum( (trace/2)**2 - det, 0 )      # discriminant

eig1 = trace/2 + np.sqrt(disc) 
eig2 = trace/2 - np.sqrt(disc)

show_images([Ixx, Iyy, Ixy, eig1, eig2, I], ["Ixx", "Iyy", "Ixy", "eig1", "eig2", "img * 0 order"], [None]*5 + ['gray'])

# generate concavity mask
mask = np.where( (eig1 < 0) & (eig2 < 0), eig1 * eig2, 0.0 )            # assign convex regions positive numbers
mask += np.where( (eig1 > 0) & (eig2 > 0), -1 * eig1 * eig2, 0.0 )      # assign concave regions negative numbers, leaving saddle points as zero

# plot mask and convolution with gaussian side by side

show_images([mask, I, img], ["concavity", "blurred", "original"], ['seismic', 'gray', 'gray'])