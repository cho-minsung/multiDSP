import cv2
import numpy as np
import math
import random

I = cv2.imread('multiDSP/hw4/city.jpg')
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
cv2.imwrite('multiDSP/hw4/city_hough_gray.jpg', gray)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
cv2.imwrite('multiDSP/hw4/city_hough_edges.jpg', edges)
hough = cv2.HoughLines(edges, rho = 1, theta = np.pi/180, threshold = 230)
for i in range(6):
    rho = hough[i][0][0]
    theta = hough[i][0][1]
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(I, pt1,pt2,color,3)
    print('line',i+1, 'is color', color, ', and rho and theta are', rho, 'and', theta)
cv2.imwrite('multiDSP/hw4/city_hough.jpg', I)

# 1. Obtain a binary edge image using any of the techniques discussed earlier in this section.
# 2. Specify subdivisions in the rho-theta-plane.
# 3. Examine the counts of the accumulator cells for high pixel concentrations.
# 4. Examine the relationship (principally for continuity) between pixels in a chosen cell.