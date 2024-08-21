import cv2 as cv
import numpy as np

img = cv.imread("downscaled_my_samples/fruits.jpg")

# pad = np.pad(img, [50, 50, 0])

# print(pad.shape)

# cv.imshow("Padded img", pad)
cv.imshow("Image", img)
cv.waitKey(0)