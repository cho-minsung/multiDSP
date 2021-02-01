from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

# Setting the input and output files.
IN_FILE = 'hw1/sample_images/jetplane.png'
OUT_FILE = 'hw1/sample_images/jetplane_32by32.png'

# Load image, store width and height into constants
pillow_img = Image.open(IN_FILE)
IMG_W, IMG_H = pillow_img.size
img = np.array(pillow_img)
new_img = np.zeros((IMG_H,IMG_W), dtype=int)

# Make a histogram for 32 by 32 blocks
for j in range(16):
    for k in range(16):
        # filling up 32x32 histogram
        histogram = np.zeros(256, dtype=int)
        for a in range(32):
            for b in range(32):
                histogram[img[a+k*32,b+j*32]] += 1
        print('histogram:',histogram)
        p256 = np.zeros(256)
        hist_out = np.zeros(256)
        for i in range(256):
            p256[i] = histogram[i]/1024
            if i == 0:
                hist_out[i] = 256 * p256[i]
            else:
                hist_out[i] = 256 * p256.sum()
        hist_out = np.round(hist_out, 0)
        print('pdf: ',p256)
        print('hist_out:',hist_out)
        for a in range(32):
            for b in range(32):
                new_img[a+k*32,b+j*32] = hist_out[img[a+k*32,b+j*32]]

output_file = Image.fromarray(np.uint8(new_img.reshape((IMG_H, IMG_W))))
output_file.save(OUT_FILE)

