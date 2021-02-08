from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Setting the input and output files.
IN_FILE = 'hw2/sample_images/Fig0627(a)(WashingtonDC Band3-RED).tiff'
OUT_FILE = 'hw2/sample_images/Fig0627_out_b.png'

# Loading the image, storing width and height into constants
# Load image, store width and height into constants
pillow_img = Image.open(IN_FILE)
IMG_W, IMG_H = pillow_img.size
img = np.array(pillow_img)

threshold_img = np.zeros((IMG_H, IMG_W), dtype=int)
for i in range(IMG_H):
    for j in range(IMG_W):    
        if (img[i,j] > 100):
            threshold_img[i, j] = 0
        else:
            threshold_img[i, j] = 255

# Saving the threshold image as .jpg
output_file = Image.fromarray(np.uint8(threshold_img.reshape((IMG_H, IMG_W))))
output_file.save(OUT_FILE)