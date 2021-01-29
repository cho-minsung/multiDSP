from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

# Setting the input and output files.
IN_FILE = 'hw1/sample_images/jetplane.png'
OUT_FILE = 'hw1/sample_images/jetplane_64.png'

# # Set the bin size
# bin = 256
# bin = 128
bin = 64

# Load image, store width and height into constants
pillow_img = Image.open(IN_FILE)
IMG_W, IMG_H = pillow_img.size
MN = IMG_H*IMG_W
img = np.array(pillow_img).flatten()

# Make a histogram
histogram = np.zeros(256, dtype=int)
for i in range(img.size):
    histogram[img[i]] += 1

# Divide the histogram into bin sizes and get the intensity distribution and transformation function.
if bin == 256:
    p256 = np.zeros(bin)
    hist_out = np.zeros(bin)
    for i in range(bin):
        p256[i] = histogram[i]/MN
        if i == 0:
            hist_out[i] = bin * p256[i]
        else:
            hist_out[i] = bin * p256.sum()
    hist_out = np.round(hist_out, 0)
elif bin == 128:
    n128 = np.zeros(bin, dtype=int)
    p128 = np.zeros(bin)
    s128 = np.zeros(bin)
    for i in range(bin):
        for j in range(int(i*256/bin), int((i+1)*256/bin)):
            n128[i] += histogram[j]
        p128[i] = n128[i]/MN
        if i == 0:
            s128[i] = bin * p128[i]
        else:
            s128[i] = bin * p128.sum()
    s128 = np.round(s128, 0)
elif bin == 64:
    n64 = np.zeros(bin, dtype=int)
    p64 = np.zeros(bin)
    s64 = np.zeros(bin)
    for i in range(bin):
        for j in range(int(i*256/bin), int((i+1)*256/bin)):
            n64[i] += histogram[j]
        p64[i] = n64[i]/MN
        if i == 0:
            s64[i] = bin * p64[i]
        else:
            s64[i] = bin * p64.sum()
    s64 = np.round(s64, 0)

# Make the new histogram for bin size 128 and 64
if bin == 128:
    hist_out = np.zeros(256, dtype = int)
    for i in range(256):
        hist_out[i]=s128[math.floor(i/2)]
elif bin == 64:
    hist_out = np.zeros(256, dtype = int)
    for i in range(256):
        hist_out[i]=s64[math.floor(i/4)]

# Make the new output image
new_img = np.zeros(img.size, dtype=int)
for i in range(img.size):
    new_img[i] = hist_out[img[i]]

# # # Save the image
output_file = Image.fromarray(np.uint8(new_img.reshape((IMG_H, IMG_W))))
output_file.save(OUT_FILE)

# # # Display the old (blue) and new (orange) histograms next to eachother
x_axis = np.arange(256)
fig = plt.figure()
fig.add_subplot(2, 2, 1)
a = plt.bar(x_axis, histogram)
a = plt.title('before')
fig.add_subplot(2, 2, 2)
b = plt.bar(x_axis, hist_out, color="orange")
b = plt.title('after')
plt.show()