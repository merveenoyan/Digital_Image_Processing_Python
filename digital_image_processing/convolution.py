# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/convolution.ipynb (unless otherwise specified).

__all__ = ['image', 'gray', 'mask', 'mask_r', 'conv']

# Cell

# Cell
import cv2 
import numpy as np
from cv2 import imread, cvtColor, COLOR_RGB2GRAY


# Cell
image = cv2.imread("lena.png")
# Cell
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# do not transpose this to get vertical change
# Cell
mask = np.array([[-1, 0, 1],[-1, 0, 1], [-1, 0, 1]]) 

# Cell
mask_r = np.transpose(mask)
# Cell
conv = cv2.filter2D(image, ddepth=-1, kernel=mask_r)
cv2.imshow("Lena Prewitt Convolution", conv)
cv2.waitKey(0)
cv2.destroyAllWindows()