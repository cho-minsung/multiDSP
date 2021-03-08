import cv2
import numpy as np
import math
import random
from PIL import Image

# IN_FILE = 'multiDSP/hw4/quarters.bmp'
# pillow_img = Image.open(IN_FILE).convert('L')
# I = np.array(pillow_img)
# rows, dummy = pillow_img.size
# output_file = Image.fromarray(np.uint8(I))
# output_file.save('multiDSP/hw4/quarters.jpg')

IN_FILE = 'multiDSP/hw4/us_silver_coins.jpg'
pillow_img = Image.open(IN_FILE).convert('L')
I = np.array(pillow_img)
rows, dummy = pillow_img.size

blur = cv2.GaussianBlur(I, ksize = (5, 5), sigmaX = 3)

edges = cv2.Canny(blur, 100, 150)
cv2.imwrite('multiDSP/hw4/us_silver_coins_edge.jpg', edges)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=30, minRadius=25, maxRadius=60)

if circles is not None:
    circles = np.uint8(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(I, center, 1, (0, 100, 100), 2)
        # circle outline
        radius = i[2]
        cv2.circle(I, center, radius, (0, 0, 255), 3)
output_file = Image.fromarray(np.uint8(I))
output_file.save('multiDSP/hw4/us_silver_coins_circle.jpg')

