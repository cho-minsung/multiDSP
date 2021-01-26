from PIL import Image
from libtiff import TIFFfile, TIFFimage
import numpy as np
from matplotlib import pyplot as plt

infile = TIFFfile.open('hw1\sample_images\jetplane.tif')
# outfile = TIFF.'hw1\sample_images\jetplane_out.png'

# def make_histogram(img):
#     histogram = np.zeros(256, dtype=int)
#     for i in range(img.size):
#         histogram[img[i]] += 1
#     return histogram

img = Image.open(infile)
img_w, img_h = img.size

print(img_h, img_w)