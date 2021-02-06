from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Setting the input and output files.
IN_FILE_RED = 'hw2/sample_images/Fig0627(a)(WashingtonDC Band3-RED).tiff'
IN_FILE_GREEN = 'hw2/sample_images/Fig0627(b)(WashingtonDC Band2-GREEN).tiff'
IN_FILE_BLUE = 'hw2/sample_images/Fig0627(a)(WashingtonDC Band1-BLUE).tiff'

# Loading the image, storing width and height into constants
