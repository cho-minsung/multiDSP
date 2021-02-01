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

# Make a box
def make_box(i, j, box):
    box[i:i+60, j:j+5, 0] = 255
    box[i:i+60, j+45:j+50, 0] = 255
    box[i:i+5, j:j+50, 0] = 255
    box[i+60:i+65, j:j+50, 0] = 255
    return box

# Setting the input and output files.
IN_FILE = 'hw1/sample_images/barcelona-team.jpg'
THRESHOLD_OUT_FILE = 'hw1/sample_images/barcelona-team_threshold.jpg'
EROSION_OUT_FILE = 'hw1/sample_images/barcelona-team_erosion.jpg'
DILATION_OUT_FILE = 'hw1/sample_images/barcelona-team_dilation.jpg'
BOXED_OUT_FILE = 'hw1/sample_images/barcelona-team_boxed.jpg'

# Loading the image and converting to array
pillow_img = Image.open(IN_FILE)
IMG_W, IMG_H = pillow_img.size
hsv_img = np.array(pillow_img.convert('HSV'))
img = np.array(pillow_img)

# Thresholding with the rule of Hue > 30 is skin
# I got this by looking at the results.
threshold_img = np.zeros((IMG_H, IMG_W), dtype=int)
for i in range(IMG_H):
    for j in range(IMG_W):    
        if hsv_img[i, j, 0] > 30:
            threshold_img[i, j] = 255
        else:
            threshold_img[i, j] = 0

# Saving the threshold image as .jpg
output_file = Image.fromarray(np.uint8(threshold_img.reshape((IMG_H, IMG_W))))
output_file.save(THRESHOLD_OUT_FILE)

# Morphology method: erosion then dilation.
# Setting the borders as is. No morphing performed with 5x5 on the borders for simplicity.
erosion_img = np.zeros((IMG_H, IMG_W), dtype=int)
erosion_img[0:1,:] = threshold_img[0:1,:]
erosion_img[IMG_H-2:IMG_H-1:] = threshold_img[IMG_H-2:IMG_H-1:]
erosion_img[:,0:1] = threshold_img[:,0:1]
erosion_img[:,IMG_W-2:IMG_W-1] = threshold_img[:,IMG_W-2:IMG_W-1]
# Performing erosion
for i in range(2, IMG_H-2):
    for j in range(2, IMG_W-2):    
        erase_flag = 0
        for k in range(-2,3):
            for l in range(-2,3):
                if threshold_img[i-k, j-l] != 0 : erase_flag = 1
        if erase_flag == 1 : erosion_img[i, j] = 255
        else: erosion_img[i, j] = 0

output_file = Image.fromarray(np.uint8(erosion_img.reshape((IMG_H, IMG_W))))
output_file.save(EROSION_OUT_FILE)

#performing dilation
dilation_img = np.zeros((IMG_H, IMG_W), dtype=int)
dilation_img[0,:] = erosion_img[0,:]
dilation_img[IMG_H-1:] = erosion_img[IMG_H-1:]
dilation_img[:,0] = erosion_img[:,0]
dilation_img[:,IMG_W-1] = erosion_img[:,IMG_W-1]
for i in range(1, IMG_H-1):
    for j in range(1, IMG_W-1):    
        create_flag = 0
        for k in range(-1,2):
            for l in range(-1,2):
                if erosion_img[i-k, j-l] == 0 : create_flag = 1
        if create_flag == 1 : dilation_img[i, j] = 0
        else: dilation_img[i, j] = 255

output_file = Image.fromarray(np.uint8(dilation_img.reshape((IMG_H, IMG_W))))
output_file.save(DILATION_OUT_FILE)

# Making the boxes using the function above.
make_box(60, 60, img)
make_box(60, 150, img)
make_box(60, 230, img)
make_box(40, 310, img)
make_box(40, 400, img)
make_box(40, 480, img)
make_box(180, 70, img)
make_box(180, 160, img)
make_box(180, 260, img)
make_box(180, 360, img)
make_box(180, 470, img)

output_file = Image.fromarray(img)
output_file.save(BOXED_OUT_FILE)

