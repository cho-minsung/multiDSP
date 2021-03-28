import numpy as np
import cv2 as cv
from PIL import Image


# initializing the arrays
background = np.zeros((120, 160), dtype=np.uint32)
for i in range(119):
    temp = cv.imread('traffic/traffic_'+str(i+1)+'.jpg')
    temp_array = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    background += temp_array
background = background / 119
background = background.astype(np.uint8)
output_file = Image.fromarray(np.uint8(background))
output_file.save('avg_traffic.jpg')

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
        image_name = 'traffic_avg_background_subtraction/traffic_avg_background_subtracted' + str(count) + '.jpg'
        output_file.save(image_name)
        source = cv.imread(image_name, cv.CV_8UC1)
        image = cv.adaptiveThreshold(source, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        output_file = Image.fromarray(np.uint8(image))
        output_file.save(image_name)
        count = count + 1
cap.release()
