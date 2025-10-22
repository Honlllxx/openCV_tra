import cv2

import cv2 as cv
import numpy as np
img = cv.imread('data/th3.png',0)
kernel = np.ones((5,5),np.uint8)
dilation = cv.dilate(img,kernel,iterations = 1)
erosion = cv.erode(img,kernel,iterations = 1)

cv2.imshow("origion", img)
cv2.imshow("dilation", dilation)
cv2.imshow("erosion", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

cv2.imshow("opening", opening)
cv2.imshow("closing", closing)
cv2.waitKey(0)

result = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
cv2.imshow("result", result)

cv2.imwrite("data/dilation.png", dilation)
cv2.imwrite("data/erosion.png", erosion)
cv2.imwrite("data/opening.png", opening)
cv2.imwrite("data/closing.png", closing)
cv2.imwrite("data/morph_result.png", result)
cv2.waitKey(0)