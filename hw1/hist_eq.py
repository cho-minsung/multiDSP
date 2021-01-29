from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Setting the input and output files.
IN_FILE = 'hw1/sample_images/jetplane.png'
# OUT_FILE = 'hw1\sample_images\jetplane_output64.png'

# # Set the bin size
# bin = 256   #8 bit
# bin = 128 #7 bit
bin = 64  #6 bit

# # make cumulative sum
# def make_cumsum(histogram):
#     cumsum = np.zeros(256, dtype=int)
#     cumsum[0] = histogram[0]
#     for i in range(1, histogram.size):
#         cumsum[i] = cumsum[i-1] + histogram[i]
#     return cumsum

# # make mapping
# def make_mapping(cumsum):
#     mapping = np.zeros(256, dtype=int)
#     grey_levels = bin
#     for i in range(grey_levels):
#         mapping[i] = max(0, round((grey_levels*cumsum[i])/(IMG_H*IMG_W))-1)
#     return mapping

# # apply mapping
# def apply_mapping(img, mapping):
#     """ Apply the mapping to our image """
#     new_image = np.zeros(img.size, dtype=int)
#     for i in range(img.size):
#         new_image[i] = mapping[img[i]]
#     return new_image

# Load image, store width and height into constants
pillow_img = Image.open(IN_FILE)
IMG_W, IMG_H = pillow_img.size
img = np.array(pillow_img).flatten()

# Make a histogram
histogram = np.zeros(256, dtype=int)
for i in range(img.size):
    histogram[img[i]] += 1

indices = np.zeros(64, dtype=int)
for i in range(64):
    indices[i] = 1 + indices[i-1]
print(indices)

# Divide the histogram into bin sizes and get the intensity distribution.
# if bin == 256: # 8 bit
#     p256 = np.zeros(8, dtype=int)
#     for i in range(8):
#         p256[i] = histogram[0:63].sum()
# elif bin == 128: # 7 bit
#     p128 = np.zeros(7, dtype=int)
#     for i in range(7):
#         for j in range(int(512*(i+1)/7)):
#             p128[i] += histogram[j-1]
# elif bin == 64: # 6 bit
#     p64 = np.zeros(6, dtype=int)
#     for i in range(6):
#         for j in range(int(512*(i+1)/6)):
#             p64[i] += histogram[j-1]

# print(p256)

# Get transformation function
# s256 = np.zeros(256, dtype=int)
# if bin == 128:
#     s128 = np.zeros(7, dtype=int)
#     for i in range(7):
        
# s64 = np.zeros(64, dtype=int)


    
# cumsum = make_cumsum(histogram)
# mapping = make_mapping(cumsum)
# new_image = apply_mapping(img, mapping)

# # Read in and flatten our greyscale image
# print('Original image:',np.array(pillow_img))
# img = np.array(pillow_img).flatten()
# print('Flattened image',np.array(img))
# print('Original image width:', IMG_W)
# print('Original image height:',IMG_H)
# print('Flatten image size: 512 * 512 = ', img.size)

# # Save the image
# output_image = Image.fromarray(np.uint8(new_image.reshape((IMG_H, IMG_W))))
# output_image.save(OUT_FILE)

# # Display the old (blue) and new (orange) histograms next to eachother
# x_axis = np.arange(256)
# fig = plt.figure()
# fig.add_subplot(2, 2, 1)
# before = mpimg.imread('hw1\sample_images\jetplane.png')
# beforeplot = plt.imshow(before, cmap = 'gray')
# fig.add_subplot(2, 2, 2)
# after = mpimg.imread('hw1\sample_images\jetplane_output64.png')
# afterplot = plt.imshow(after, cmap = 'gray')
# fig.add_subplot(2, 2, 3)
# a = plt.bar(x_axis, histogram)
# a = plt.title('before')
# fig.add_subplot(2, 2, 4)
# b = plt.bar(x_axis, make_histogram(new_image), color="orange")
# b = plt.title('after')
# plt.show()