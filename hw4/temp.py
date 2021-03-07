import cv2
import numpy as np
from scipy import signal as sp
from matplotlib import pyplot as plt
from PIL import Image

sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

IN_FILE = 'hw4/city.jpg'
pillow_img = Image.open(IN_FILE).convert('L')
IMG_W, IMG_H = pillow_img.size
I = np.array(pillow_img)
# output_file = Image.fromarray(np.uint8(I))
# output_file.save('hw4/wirebond_mask.jpg')
s1 = 3
tau = 20

# 1. Smooth the input image with a Gaussian filter.
blur = cv2.GaussianBlur(I, ksize = (5, 5), sigmaX = s1)
output_file = Image.fromarray(np.uint8(blur))
output_file.save('hw4/city_blur.jpg')

# 2. Compute the gradient magnitude and angle images.
g_x = sp.convolve2d(blur, sobel_x)
g_y = sp.convolve2d(blur, sobel_y)
output_file = Image.fromarray(np.uint8(g_x))
output_file.save('hw4/city_sobel_x.jpg')
output_file = Image.fromarray(np.uint8(g_y))
output_file.save('hw4/city_sobel_y.jpg')
M = np.sqrt(np.square(g_x) + np.square(g_y))
M *= 255 / M.max()
output_file = Image.fromarray(np.uint8(M))
output_file.save('hw4/city_M.jpg')
A = np.arctan(g_y / g_x)
# plt.plot(A)
# plt.show()

# 3. Apply nonmaxima suppression to the gradient magnitude image.
ret, E = cv2.threshold(M,tau, 255, cv2.THRESH_BINARY_INV)
output_file = Image.fromarray(np.uint8(E))
output_file.save('hw4/city_E_20_3.jpg')

# default canny edge
default = cv2.Canny(I, 100, 200)
output_file = Image.fromarray(np.uint8(default))
output_file.save('hw4/city_default_canny.jpg')

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