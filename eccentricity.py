# this file demonstrates use of hessian matrix eigenvalues to measure tissue eccentricty

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from utils.geometry import hessian_eigenvalues
from utils.basic import read_image_grayscale
from utils.visualization import show_images


filepath = 'images/ventricles/vent1.png'
sigma = 20

# load image as grayscale
img = read_image_grayscale(filepath)

# visualize convolution with gaussian
I = gaussian_filter(img, sigma=sigma, order=(0, 0))

# compute eigenvalues 
eig1, eig2 = hessian_eigenvalues(img, sigma)

# generate eccentricity mask
mask = (eig1 - eig2) / (eig1 + eig2)

# clip top and bottom 1% of values
p_low, p_high = np.percentile(mask, [1, 99])
mask = np.clip(mask, p_low, p_high)

# plot mask and convolution with gaussian side by side
show_images([mask, I, img], ['eccentricity mask', 'img * 0th order', 'raw image'], ['seismic', 'gray', 'gray'])