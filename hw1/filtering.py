from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Setting the input and output files.
IN_FILE = '/home/minsung/multiDSP/hw1/sample_images/DSCN0479-001.jfif'
# OUT_FILE = 'hw1\sample_images\jetplane_output64.png'

# Load image, store width and height into constants
pillow_img = Image.open(IN_FILE)
IMG_W, IMG_H = pillow_img.size

img = np.array(pillow_img)

Y = np.zeros((IMG_H,IMG_W))
Cb = np.zeros((IMG_H,IMG_W))
Cr = np.zeros((IMG_H,IMG_W))

for i in range(IMG_H):
    for j in range(IMG_W):
        Y[i, j] = img[i, j, 0]
        Cb[i, j] = img[i, j, 1]
        Cr[i, j] = img[i, j, 2]

print(Y)
print(Cb)
print(Cr)



