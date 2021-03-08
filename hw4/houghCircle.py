import cv2
import numpy as np
import math
import random
from PIL import Image

IN_FILE = 'multiDSP/hw4/quarters.bmp'
pillow_img = Image.open(IN_FILE).convert('L')
I = np.array(pillow_img)
IMG_W, IMG_H = pillow_img.size
output_file = Image.fromarray(np.uint8(I))
output_file.save('multiDSP/hw4/quarters.jpg')

edges = cv2.Canny(I, 100, 200)
cv2.imwrite('multiDSP/hw4/quarters_edges.jpg', edges)

# Voting method. It did not work because it is computationally demanding.
A = np.zeros((2, IMG_H*IMG_W))
r = 20
for x in range(IMG_H):
    for y in range(IMG_W):
        for t in range(360):
            if int(x - r * np.cos(t * np.pi / 180)) < 0 or int(y - r * np.sin(t * np.pi / 180)) < 0:
                continue
            elif int(x - r * np.cos(t * np.pi / 180)) >= IMG_H or int(y - r * np.sin(t * np.pi / 180)) >= IMG_W:
                continue
            elif edges[int(x - r * np.cos(t * np.pi / 180)), int(y - r * np.sin(t * np.pi / 180))] == 255:
                A[x][y] += 1

