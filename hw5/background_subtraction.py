import numpy as np
import cv2 as cv
from PIL import Image


# initializing the arrays
difference = np.array([])
background = cv.imread('traffic_119.jpg')
background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
output_file = Image.fromarray(np.uint8(background))
output_file.save('traffic_background.jpg')
image = 0
cap = cv.VideoCapture('traffic.avi')
count = 1
flag = 1

while flag:
    flag, frame = cap.read()
    if flag:
        present = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        difference = np.subtract(present, background, dtype=np.int8)
        difference[difference < 0] = 0
        output_file = Image.fromarray(np.uint8(difference))
        image_name = 'traffic_background_subtraction/traffic_background_subtracted' + str(count) + '.jpg'
        output_file.save(image_name)
        source = cv.imread(image_name, cv.CV_8UC1)
        image = cv.adaptiveThreshold(source, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        output_file = Image.fromarray(np.uint8(image))
        output_file.save(image_name)
        count = count + 1
cap.release()
