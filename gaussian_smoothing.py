import cv2 
import numpy as np
from cv2 import imread, cvtColor, COLOR_RGB2GRAY
import math



image = cv2.imread("lena.png")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

size = size # enter a value here, will be taken with argparse later
g = np.zeros((size, size))
sigma = sigma # same with size

for i in range(-math.floor(size/2), math.floor(size/2)):

    for j in range(-math.floor(size/2), math.floor(size/2)):

        g[i+math.floor(size / 2), j+math.floor(size/2)] = (1/(2*math.pi*sigma^2)) * np.exp(-(i^2+j^2)/(2*sigma^2))

g = g/sum(sum(g))
conv = cv2.filter2D(image, ddepth=-1, kernel=g)
cv2.imshow("Gaussian Smoothing", conv)
cv2.waitKey(0)
