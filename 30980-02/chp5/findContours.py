import cv2

img = cv2.imread('data/morph_result.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow("img", img)
cv2.waitKey(0)

cv2.imwrite("data/morph_contour.png", img)
