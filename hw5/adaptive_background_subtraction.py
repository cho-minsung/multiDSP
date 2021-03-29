import numpy as np
import cv2 as cv
from PIL import Image


image = 0
cap = cv.VideoCapture('traffic.avi')
count = 1
flag = 1
backSub = cv.createBackgroundSubtractorMOG2(history=100)

while flag:
    flag, frame = cap.read()
    if flag:
        fgMask = backSub.apply(frame)
        image_name = 'traffic_adaptive_background/traffic_adaptive_background' + str(count) + '.jpg'
        output_file = Image.fromarray(np.uint8(fgMask))
        output_file.save(image_name)
        count = count + 1
cap.release()
