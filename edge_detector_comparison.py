'''
this file will compare the following edge detection methods
1st order gaussian
sobel
discrete laplacian                  (gives concavity info too)
2nd order gaussian -> laplacian     (gives concavity info too)
'''

from utils.basic import read_image_grayscale
from utils.basic import normalize
from utils.geometry import hessian_eigenvalues
from utils.visualization import show_images
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
import torch.nn.functional as F
import numpy as np
import torch


# user defined
filepath = 'images/basic/circle_square.png'
sigma = 1

# import image 
img = read_image_grayscale(filepath)

# 1st order gaussian
Gx = gaussian_filter(img, sigma, order=(0, 1))
Gy = gaussian_filter(img, sigma, order=(1, 0))
gradient_gaussian = np.hypot(Gx, Gy)

# sobel edge detection
Ix = sobel(img, axis=1)
Iy = sobel(img, axis=0)
gradient_sobel = np.hypot(Ix, Iy)

# 2nd order gaussian (LoG)
eig1, eig2 = hessian_eigenvalues(img, sigma)
laplacian_gaussian = eig1 + eig2

# discrete laplacian
dl_kernel = torch.tensor([  [ 0,  1,  0],
                            [ 1, -4,  1],
                            [ 0,  1,  0]], dtype=torch.float32).view(1, 1, 3, 3)

img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
laplacian_discrete = F.conv2d(img_tensor, dl_kernel, stride=1, padding=1).squeeze()

# normalize all
images = [gradient_gaussian, gradient_sobel, laplacian_gaussian, laplacian_discrete]

images_norm = [normalize(i) for i in images]

# visualize
# [ADD] normalization
show_images( images_norm, ['gradient: gaussian', 'gradient_sobel', 'laplacian: gaussian', 'laplacian: discrete'])