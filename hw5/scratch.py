import numpy as np
import cv2 as cv
from PIL import Image


# initializing the arrays
present = np.array([])
previous = np.array([])
difference = np.array([])

cap = cv.VideoCapture('traffic.avi')
flag, frame = cap.read()
previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
count = 1
image_shape = previous.shape
fps = 15
image = 0

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('traffic_diff.avi', fourcc, fps, image_shape, 0)

while flag:
    flag, frame = cap.read()
    if flag:
        present = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        output_file = Image.fromarray(np.uint8(present))
        orig_image_name = 'traffic/orig_traffic_' + str(count) + '.jpg'
        output_file.save(orig_image_name)
        difference = np.subtract(present, previous, dtype=np.int8)
        np.absolute(difference, difference)
        previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        image_name = 'traffic_diff/traffic_' + str(count) + '.jpg'
        output_file = Image.fromarray(np.uint8(difference))
        output_file.save(image_name)
        source = cv.imread(image_name, cv.CV_8UC1)
        image = cv.adaptiveThreshold(source, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        output_file = Image.fromarray(np.uint8(image))
        output_file.save(image_name)
        count = count + 1
cap.release()
