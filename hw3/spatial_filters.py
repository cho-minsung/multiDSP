from PIL import Image
import numpy as np
import math

IN_FILE = 'hw3/city.jpg'

pillow_img = Image.open(IN_FILE).convert('L')
pillow_img.save('hw3/city_grayscale.jpg')
IMG_W, IMG_H = pillow_img.size
img = np.array(pillow_img)

# output_img = np.zeros(shape=(IMG_H, IMG_W), dtype=np.int8)
# output_img[0:3,:] = img[0:3,:]
# output_img[IMG_H-3:IMG_H,:] = img[IMG_H-3:IMG_H:]
# output_img[:,0:3] = img[:,0:3]
# output_img[:,IMG_W-3:IMG_W] = img[:,IMG_W-3:IMG_W]

# temp = 0
# for i in range(3, IMG_H-3):
#     for j in range(3, IMG_W-3):
#         for k in range(7):
#             for l in range(7):
#                 temp += img[i+3-k, j+3-l]
#         output_img[i, j] = temp / (7 * 7)
#         temp = 0        

# output_file = Image.fromarray(np.uint8(output_img))
# output_file.save('hw3/city_output.jpg')

# Gaussian kernel with sigma of 0.5.
gaussian_point5 = np.zeros(shape=(7,7), dtype=float)
# 2nd distribution
gaussian_point5[1,1] = 0.000002
gaussian_point5[1,5] = 0.000002
gaussian_point5[5,1] = 0.000002
gaussian_point5[5,5] = 0.000002
gaussian_point5[2,1] = 0.000212
gaussian_point5[4,1] = 0.000212
gaussian_point5[1,2] = 0.000212
gaussian_point5[5,2] = 0.000212
gaussian_point5[1,4] = 0.000212
gaussian_point5[5,4] = 0.000212
gaussian_point5[2,5] = 0.000212
gaussian_point5[4,5] = 0.000212
gaussian_point5[3,1] = 0.000922
gaussian_point5[1,3] = 0.000922
gaussian_point5[5,3] = 0.000922
gaussian_point5[3,5] = 0.000922
# 1st distribution
gaussian_point5[2,2] = 0.024745
gaussian_point5[4,2] = 0.024745
gaussian_point5[2,4] = 0.024745
gaussian_point5[4,4] = 0.024745
gaussian_point5[3,2] = 0.10739
gaussian_point5[2,3] = 0.10739
gaussian_point5[4,3] = 0.10739
gaussian_point5[3,4] = 0.10739
# Center
gaussian_point5[3,3] = 0.466064

# Gaussian kernel with sigma of 3.
gaussian_3 = np.zeros(shape=(7,7), dtype=float)
# 3rd distribution
gaussian_3[0,0] = 0.011362
gaussian_3[0,6] = 0.011362
gaussian_3[6,0] = 0.011362
gaussian_3[6,6] = 0.011362
gaussian_3[0,1] = 0.014962
gaussian_3[0,5] = 0.014962
gaussian_3[1,0] = 0.014962
gaussian_3[1,6] = 0.014962
gaussian_3[5,0] = 0.014962
gaussian_3[5,6] = 0.014962
gaussian_3[6,1] = 0.014962
gaussian_3[6,5] = 0.014962
gaussian_3[0,2] = 0.017649
gaussian_3[0,4] = 0.017649
gaussian_3[2,0] = 0.017649
gaussian_3[2,6] = 0.017649
gaussian_3[4,0] = 0.017649
gaussian_3[4,6] = 0.017649
gaussian_3[6,2] = 0.017649
gaussian_3[6,4] = 0.017649
gaussian_3[3,0] = 0.018648
gaussian_3[0,3] = 0.018648
gaussian_3[3,6] = 0.018648
gaussian_3[6,3] = 0.018648
# 2nd distribution
gaussian_3[1,1] = 0.019703
gaussian_3[1,5] = 0.019703
gaussian_3[5,1] = 0.019703
gaussian_3[5,5] = 0.019703
gaussian_3[2,1] = 0.02324
gaussian_3[4,1] = 0.02324
gaussian_3[1,2] = 0.02324
gaussian_3[5,2] = 0.02324
gaussian_3[1,4] = 0.02324
gaussian_3[5,4] = 0.02324
gaussian_3[2,5] = 0.02324
gaussian_3[4,5] = 0.02324
gaussian_3[3,1] = 0.024556
gaussian_3[1,3] = 0.024556
gaussian_3[5,3] = 0.024556
gaussian_3[3,5] = 0.024556
# 1st distribution
gaussian_3[2,2] = 0.027413
gaussian_3[4,2] = 0.027413
gaussian_3[2,4] = 0.027413
gaussian_3[4,4] = 0.027413
gaussian_3[3,2] = 0.028964
gaussian_3[2,3] = 0.028964
gaussian_3[4,3] = 0.028964
gaussian_3[3,4] = 0.028964
# Center
gaussian_3[3,3] = 0.030603

# gauss_point5_img = np.zeros(shape=(IMG_H, IMG_W), dtype=np.int8)
# gauss_point5_img[0:3,:] = img[0:3,:]
# gauss_point5_img[IMG_H-3:IMG_H,:] = img[IMG_H-3:IMG_H:]
# gauss_point5_img[:,0:3] = img[:,0:3]
# gauss_point5_img[:,IMG_W-3:IMG_W] = img[:,IMG_W-3:IMG_W]

# for i in range(3, IMG_H-3):
#     for j in range(3, IMG_W-3):
#         for k in range(7):
#             for l in range(7):
#                 gauss_point5_img[i, j] += int(img[i+3-k, j+3-l] * gaussian_point5[k, l])

# output_file = Image.fromarray(np.uint8(gauss_point5_img))
# output_file.save('hw3/city_gauss_point5_output.jpg')

gauss_3_img = np.zeros(shape=(IMG_H, IMG_W), dtype=np.int8)
gauss_3_img[0:3,:] = img[0:3,:]
gauss_3_img[IMG_H-3:IMG_H,:] = img[IMG_H-3:IMG_H:]
gauss_3_img[:,0:3] = img[:,0:3]
gauss_3_img[:,IMG_W-3:IMG_W] = img[:,IMG_W-3:IMG_W]

for i in range(3, IMG_H-3):
    for j in range(3, IMG_W-3):
        for k in range(7):
            for l in range(7):
                gauss_3_img[i, j] += int(img[i+3-k, j+3-l] * gaussian_3[k, l])

# output_file = Image.fromarray(np.uint8(gauss_3_img))
# output_file.save('hw3/city_gauss_3_output.jpg')

# sobelx_kern = np.zeros(shape=(3,3), dtype = np.int16)
# sobelx_kern[0,0]=1
# sobelx_kern[0,2]=1
# sobelx_kern[0,1]=2
# sobelx_kern[2,0]=-1
# sobelx_kern[2,2]=-1
# sobelx_kern[2,1]=-2

# sobely_kern = np.zeros(shape=(3,3), dtype = np.int16)
# sobely_kern[0,0]=1
# sobely_kern[2,0]=1
# sobely_kern[1,0]=2
# sobely_kern[0,2]=-1
# sobely_kern[2,2]=-1
# sobely_kern[1,2]=-2

# sobel_img = np.zeros(shape=(IMG_H, IMG_W), dtype = np.int16)
# sobel_img[0:1,:] = img[0:1,:]
# sobel_img[IMG_H-1:IMG_H,:] = img[IMG_H-1:IMG_H:]
# sobel_img[:,0:1] = img[:,0:1]
# sobel_img[:,IMG_W-1:IMG_W] = img[:,IMG_W-1:IMG_W]

# sobelx = 0
# sobely = 0
# for i in range(1, IMG_H-1):
#     for j in range(1, IMG_W-1):
#         for k in range(3):
#             for l in range(3):
#                 sobelx += img[i-k+1, j-l+1] * sobelx_kern[k, l]
#                 sobely += img[i-k+1, j-l+1] * sobely_kern[k, l]
#         sobel_img[i, j] = math.sqrt(sobelx**2+sobely**2)
#         sobelx = 0
#         sobely = 0

# output_file = Image.fromarray(np.uint8(sobel_img))
# output_file.save('hw3/city_sobel_output.jpg')

# laplacian_img = np.zeros(shape=(IMG_H, IMG_W), dtype = np.int8)
# laplacian_img[0:1,:] = img[0:1,:]
# laplacian_img[IMG_H-1:IMG_H,:] = img[IMG_H-1:IMG_H:]
# laplacian_img[:,0:1] = img[:,0:1]
# laplacian_img[:,IMG_W-1:IMG_W] = img[:,IMG_W-1:IMG_W]

# first_der = np.zeros(shape=(IMG_H, IMG_W, 4), dtype = np.int8)

# for i in range(1, IMG_H-1):
#     for j in range(1, IMG_W-1):
#             first_der[i, j, 0] = int((img[i,j+1]-img[i,j-1])/2)
#             first_der[i, j, 1] = int((img[i+1,j]-img[i-1,j])/2)
#             first_der[i, j, 2] = int((img[i+1,j+1]-img[i-1,j-1])/2)
#             first_der[i, j, 3] = int((img[i-1,j+1]-img[i+1,j-1])/2)

# for i in range(1, IMG_H-1):
#     for j in range(1, IMG_W-1):
#             if (first_der[i,j+1, 0]-first_der[i,j-1,0])/2 == 0 or (first_der[i+1,j, 1]-first_der[i-1,j, 1])/2 == 0 or ((first_der[i+1,j+1, 2]-first_der[i-1,j-1, 2])/2) == 0 or ((first_der[i-1,j+1, 3]-first_der[i+1,j-1, 3])/2) == 0:
#                 laplacian_img[i, j] = 255
#             else:
#                 laplacian_img[i, j] = 0

# output_file = Image.fromarray(np.uint8(laplacian_img))
# output_file.save('hw3/city_laplacian_output.jpg')

gaussian_laplacian_img = np.zeros(shape=(IMG_H, IMG_W), dtype = np.int8)
gaussian_laplacian_img[0:1,:] = img[0:1,:]
gaussian_laplacian_img[IMG_H-1:IMG_H,:] = img[IMG_H-1:IMG_H:]
gaussian_laplacian_img[:,0:1] = img[:,0:1]
gaussian_laplacian_img[:,IMG_W-1:IMG_W] = img[:,IMG_W-1:IMG_W]

first_der = np.zeros(shape=(IMG_H, IMG_W, 4), dtype = np.int8)

for i in range(1, IMG_H-1):
    for j in range(1, IMG_W-1):
            first_der[i, j, 0] = int((gauss_3_img[i,j+1]-gauss_3_img[i,j-1])/2)
            first_der[i, j, 1] = int((gauss_3_img[i+1,j]-gauss_3_img[i-1,j])/2)
            first_der[i, j, 2] = int((gauss_3_img[i+1,j+1]-gauss_3_img[i-1,j-1])/2)
            first_der[i, j, 3] = int((gauss_3_img[i-1,j+1]-gauss_3_img[i+1,j-1])/2)

for i in range(1, IMG_H-1):
    for j in range(1, IMG_W-1):
            if (first_der[i,j+1, 0]-first_der[i,j-1,0])/2 == 0 or (first_der[i+1,j, 1]-first_der[i-1,j, 1])/2 == 0 or ((first_der[i+1,j+1, 2]-first_der[i-1,j-1, 2])/2) == 0 or ((first_der[i-1,j+1, 3]-first_der[i+1,j-1, 3])/2) == 0:
                gaussian_laplacian_img[i, j] = 255
            else:
                gaussian_laplacian_img[i, j] = 0

output_file = Image.fromarray(np.uint8(gaussian_laplacian_img))
output_file.save('hw3/city_gaussian_laplacian_output.jpg')