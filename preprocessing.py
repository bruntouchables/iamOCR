""" Created by Henrikh Kantuni on 12/4/16 """

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

# read an image
test_image = cv.imread('test.jpg', 0)

# show an image
cv.imshow('Original', test_image)
cv.waitKey(0)

# clean up
cv.destroyAllWindows()
