import cv2
import numpy as np
from scipy import signal as sp
from matplotlib import pyplot as plt
from PIL import Image

def canny(I, sig, tau):

    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 1. Smooth the input image with a Gaussian filter.
    blur = cv2.GaussianBlur(I, ksize = (5, 5), sigmaX = sig)

    # 2. Compute the gradient magnitude and angle images.
    g_x = sp.convolve2d(blur, sobel_x)
    g_y = sp.convolve2d(blur, sobel_y)
    M = np.sqrt(np.square(g_x) + np.square(g_y))
    M *= 255 / M.max()
    A = np.arctan(g_y / g_x)

    # 3. Apply nonmaxima suppression to the gradient magnitude image.
    ret, E = cv2.threshold(M,tau, 255, cv2.THRESH_BINARY_INV)

    return E, M, A

def hystThresh(I, tau_l, tau_h):
    tau_l=0.05, tau_h=0.09
    highThreshold = I.max() * tau_l
    lowThreshold = highThreshold * tau_h

    m, n = I.shape
    res = np.zeros((m,n), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(I >= highThreshold)
    zeros_i, zeros_j = np.where(I < lowThreshold)

    weak_i, weak_j = np.where((I <= highThreshold) & (I >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

def main():
    IN_FILE = 'hw4/wirebond_mask.tif'
    pillow_img = Image.open(IN_FILE)
    IMG_W, IMG_H = pillow_img.size
    I = np.array(pillow_img)
    sig = 0.5
    tau = 0.8

    E, M, A = canny(I, sig, tau)


if __name__ == "__main__":
    main()