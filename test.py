import cv2 as cv
import numpy as np

img = cv.imread("my_samples/fruits.png")

pad = np.pad(img, [50, 50, 0])

print(pad.shape)

# cv.imshow("Padded img", pad)

# cv.waitKey(0)