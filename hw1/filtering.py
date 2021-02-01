from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import random
import math
import time

# Setting the input and output files.
IN_FILE = 'hw1/sample_images/DSCN0482-001.jfif'
JPG_OUT_FILE = 'hw1/sample_images/DSCN0482-001.jpg'
NOISY_OUT_FILE = 'hw1/sample_images/DSCN0482-001_noisy.jpg'
SEASONED_OUT_FILE = 'hw1/sample_images/DSCN0482-001_seasoned.jpg'
THREEBOX_OUT_FILE = 'hw1/sample_images/DSCN0482-001_threebox.jpg'
SEVENBOX_OUT_FILE = 'hw1/sample_images/DSCN0482-001_sevenbox.jpg'
MEDIAN_OUT_FILE = 'hw1/sample_images/DSCN0482-001_median.jpg'


# Load image, store width and height into constants
pillow_img = Image.open(IN_FILE)
IMG_W, IMG_H = pillow_img.size
img = np.array(pillow_img)

# Make Gaussian noise with the variant of 0.005
noise = np.random.randint(255,size=(IMG_H, IMG_W, 3))
noisy_img = np.array(img + math.sqrt(0.005)*noise, dtype = int)

# # Save the images to compare
output_file = Image.fromarray(np.uint8(img))
output_file.save(JPG_OUT_FILE)
# output_file = Image.fromarray(np.uint8(noisy_img))
# output_file.save(NOISY_OUT_FILE)

# Make salt and pepper noise
# seasoned_img = img
# for i in range(1, IMG_H):
#     for j in range(1, IMG_W):
#         if np.random.randint(20) == 1:
#             seasoned_img[i, j] = [255,255,255]
#         elif np.random.randint(20) == 2:
#             seasoned_img[i, j] = [0,0,0]
# output_file = Image.fromarray(np.uint8(seasoned_img))
# output_file.save(SEASONED_OUT_FILE)

start_time = time.time()
# Smoothing with a 3x3 box filter
threebythree_box_img = np.zeros((IMG_H, IMG_W, 3), dtype=int)
threebythree_box_img[0,:] = img[0,:]
threebythree_box_img[IMG_H-1:] = img[IMG_H-1:]
threebythree_box_img[:,0] = img[:,0]
threebythree_box_img[:,IMG_W-1] = img[:,IMG_W-1]
for i in range(1, IMG_H-1):
    for j in range(1, IMG_W-1):
        for k in range(3):
            sum = 0
            for l in range(-1,2):
                for m in range(-1,2):
                    sum += img[i+l, j+m, k]
            threebythree_box_img[i, j, k] = sum / 9
print("For 3x3: --- %s seconds ---" % (time.time() - start_time))
output_file = Image.fromarray(np.uint8(threebythree_box_img))
output_file.save(THREEBOX_OUT_FILE)

# Smoothing with a 7x7 box filter
start_time = time.time()
sevenbyseven_box_img = np.zeros((IMG_H, IMG_W, 3), dtype=int)
sevenbyseven_box_img[0:2,:] = img[0:2,:]
sevenbyseven_box_img[IMG_H-3:IMG_H-1:] = img[IMG_H-3:IMG_H-1:]
sevenbyseven_box_img[:,0:2] = img[:,0:2]
sevenbyseven_box_img[:,IMG_W-3:IMG_W-1] = img[:,IMG_W-3:IMG_W-1]
for i in range(3, IMG_H-3):
    for j in range(3, IMG_W-3):
        for k in range(3):
            sum = 0
            for l in range(-3,4):
                for m in range(-3,4):
                    sum += img[i-l, j-m, k]
            sevenbyseven_box_img[i, j, k] = sum/49
print("For 7x7: --- %s seconds ---" % (time.time() - start_time))
output_file = Image.fromarray(np.uint8(sevenbyseven_box_img))
output_file.save(SEVENBOX_OUT_FILE)

# Smoothing with a median filter
start_time = time.time()
median_img = np.zeros((IMG_H, IMG_W, 3), dtype=int)
median_img[0,:] = img[0,:]
median_img[IMG_H-1:] = img[IMG_H-1:]
median_img[:,0] = img[:,0]
median_img[:,IMG_W-1] = img[:,IMG_W-1]
for i in range(1, IMG_H-1):
    for j in range(1, IMG_W-1):
        for k in range(3):
            median_array = np.zeros(9)
            count = 0
            for l in range(-1,2):
                for m in range(-1,2):
                    median_array[count] = img[i-l, j-m, k]
                    count += 1
            median_array = sorted(median_array)
            median_img[i, j, k] = median_array[4]
print("For median: --- %s seconds ---" % (time.time() - start_time))
output_file = Image.fromarray(np.uint8(median_img))
output_file.save(MEDIAN_OUT_FILE)

# Calculate the MSE for 3x3
MSE = 0
img1 = np.array(img).flatten()
img2 = np.array(threebythree_box_img).flatten()
for i in range(img.size):
    MSE += (img1[i] - img2[i])**2/(IMG_H*IMG_W*3)
print('MSE of 3x3:',MSE)
PSNR = 20*math.log(255/math.sqrt(MSE),10)
print('PSNR of 3x3:',PSNR)

# Calculate the MSE for 7x7
MSE = 0
img3 = np.array(sevenbyseven_box_img).flatten()
for i in range(img.size):
    MSE += (img1[i] - img3[i])**2/(IMG_H*IMG_W*3)
print('MSE of 7x7:',MSE)
PSNR = 20*math.log(255/math.sqrt(MSE),10)
print('PSNR of 7x7:',PSNR)

# Calculate the MSE for median
MSE = 0
img4 = np.array(median_img).flatten()
for i in range(img.size):
    MSE += (img1[i] - img4[i])**2/(IMG_H*IMG_W*3)
print('MSE of median:',MSE)
PSNR = 20*math.log(255/math.sqrt(MSE),10)
print('PSNR of median:',PSNR)