from PIL import Image
import numpy as np

IN_FILE = 'hw3/city.jpg'

pillow_img = Image.open(IN_FILE).convert('L')
IMG_W, IMG_H = pillow_img.size
img = np.array(pillow_img)

zero_padded = np.zeros(shape=(IMG_H*2, IMG_W*2), dtype=np.int8)
for i in range(IMG_H):
    for j in range(IMG_W):
        zero_padded[i, j] = img[i, j]

output_file = Image.fromarray(np.uint8(zero_padded))
output_file.save('hw3/city_zero_padded.jpg')

centered_transform = zero_padded
for i in range(IMG_H*2):
    for j in range(IMG_W*2):
        centered_transform[i, j] = zero_padded[i, j] * (-1)**(i+j)

output_file = Image.fromarray(np.uint8(centered_transform))
output_file.save('hw3/city_centered_transform.jpg')

dft = np.zeros(shape=(IMG_H*2, IMG_W*2), dtype=np.int8)
for i in range(IMG_H*2):
    for j in range(IMG_W*2):
        dft[i, j] = center[i, j] * (-1)**(i+j)