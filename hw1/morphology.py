from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Make histogram from image
def make_histogram(img):
    histogram = np.zeros(np.amax(img))
    for i in range(img.size):
        histogram[img[i]] += 1
    return histogram

# Setting the input and output files.
IN_FILE = '/home/oleggar/Minsung/multiDSP/hw1/sample_images/SJEarthquakesteampic.jpg'
OUT_FILE = '/home/oleggar/Minsung/multiDSP/hw1/sample_images/SJEarthquakesteampic_output.jpg'

# Loading the image and converting to array
pillow_img = Image.open(IN_FILE)
IMG_W, IMG_H = pillow_img.size
pillow_img = np.array(pillow_img)

# Initializing the grayscale image
gray_img = np.zeros((IMG_H, IMG_W), dtype=int)
G_IMG_H, G_IMG_W = gray_img.shape

# converting to grayscale image
# New grayscale image = ( (0.3 * R) + (0.59 * G) + (0.11 * B) )
# https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm
for i in range(G_IMG_H):
    for j in range(G_IMG_W):
        gray_img[i,j]=0.3*pillow_img[i,j,0]
        gray_img[i,j]+=0.59*pillow_img[i,j,1]
        gray_img[i,j]+=0.11*pillow_img[i,j,2]

# Flattening the grayscale image
img = np.array(gray_img).flatten()
print('Flattened image',np.array(gray_img))
print('Flatten image size:', img.size)

# Making the histogram
histogram = make_histogram(gray_img)
print('Flattened image',np.array(histogram))
