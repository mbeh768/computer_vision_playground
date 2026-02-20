# this file demonstrates use of hessian matrix eigenvalues to find concavity at a surface

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


filepath = 'circle_square.png'
img = cv2.imread(filepath)

# turn image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.astype(np.float32)

# compute second derivative approximations via gaussian filter
sigma = 25

# visualize convolution with gaussian
I = gaussian_filter(img, sigma=sigma, order=(0, 0))
figure0, axes0 = plt.subplots(1,2)

# show blurred image
axes0[0].imshow(img, cmap='gray')
axes0[0].set_title("raw image")
axes0[1].imshow(I, cmap='gray')
axes0[1].set_title("I")

# compute first derivatives
Ix = gaussian_filter(img, sigma=sigma, order=(0, 1))
Iy = gaussian_filter(img, sigma=sigma, order=(1, 0))

# show gradient
figure1, axes1 = plt.subplots(1,2)
axes1[0].imshow(Ix)
axes1[0].set_title("Ix")
axes1[1].imshow(Iy)
axes1[1].set_title("Iy")
plt.show()

# compute second derivatives
Ixx = gaussian_filter(img, sigma=sigma, order=(0, 2))
Iyy = gaussian_filter(img, sigma=sigma, order=(2, 0))
Ixy = gaussian_filter(img, sigma=sigma, order=(1, 1))

# compute eigenvalues of 2x2 hessian for each pixel
trace = Ixx + Iyy
det = Ixx * Iyy - Ixy**2

eig1 = trace/2 + np.sqrt((trace/2)**2 - det) 
eig2 = trace/2 - np.sqrt((trace/2)**2 - det)

# show second derivatives / eigenavalues
figure2, axes2 = plt.subplots(2, 3)
axes2[0, 0].imshow(Ixx)
axes2[0, 0].set_title("Ixx")
axes2[0, 1].imshow(Iyy)
axes2[0, 1].set_title("Iyy")
axes2[0, 2].imshow(Ixy)
axes2[0, 2].set_title("Ixy")
axes2[1, 0].imshow(eig1)
axes2[1, 0].set_title("eig1")
axes2[1, 1].imshow(eig2)
axes2[1, 1].set_title("eig2")
axes2[1, 2].imshow(I, cmap='gray')
axes2[1, 2].set_title("convolution with gaussian")
plt.show()

# generate concavity mask
# concavity is defined as such because a dark tissue with curvature about a brightbackground should have negative eigenvectors
mask = np.where( (eig1 < 0) & (eig2 < 0), eig1 * eig2, 0.0 )            # assign convex regions positive numbers
mask += np.where( (eig1 > 0) & (eig2 > 0), -1 * eig1 * eig2, 0.0 )      # assign concave regions negative numbers, leaving saddle points as zero

# plot mask and convolution with gaussian side by side
figure3, axes3 = plt.subplots(1, 3)
axes3[0].imshow(mask, cmap='seismic')
axes3[0].set_title('concavity mask')
axes3[1].imshow(I, cmap='gray')
axes3[1].set_title('convolution with gaussian')
axes3[2].imshow(img, cmap='gray')
axes3[2].set_title('raw iamge')
plt.show()