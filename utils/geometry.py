'''
This file houses methods for detecting geometry of objects
'''

from scipy.ndimage import gaussian_filter
import numpy as np

'''
Args
img (np.array) (np.float32):    grayscale image
sigma:                          standard deviation of gaussian
'''

def hessian_eigenvalues(img, sigma):
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

    return eig1, eig2