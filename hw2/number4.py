from PIL import Image
import matplotlib.image as mpimg
import numpy as np

original = np.zeros((7, 5), dtype=int)
original[1,2] = 1
original[2,2] = 1
original[3,3] = 1
original[3,4] = 1
original[4,3] = 1
original[5,3] = 1
original[5,1] = 2

for i in range(1,6):
    for j in range(1,5):
        if j < 4:
            print(original[i,j], end=" ")
        else:
            print(original[i,j])
print('original image; 1 represents a filled block and 2 represents the x-point.')

output = np.zeros((7, 5), dtype=int)
for i in range(1,6):
    for j in range(1, 5):
        if original[i,j]==2:
            output[i,j]=2
        if (original[i-1,j] == 1) or (original[i-1,j-1] == 1) or (original[i+1, j] == 1):
                output[i,j] = 1

for i in range(1,6):
    for j in range(1,5):
        if j < 4:
            print(output[i,j], end=" ")
        else:
            print(output[i,j])

print('output image')