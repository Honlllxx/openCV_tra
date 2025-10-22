import cv2

# 自动阈值分割
img = cv2.imread('data/155.bmp', 0)
#ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img, (3, 3), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
diff = cv2.subtract(th2, th3)
cv2.imshow("11", img)
cv2.imshow("22", th2)
cv2.imshow("33", th3)
cv2.imshow("44", diff)
cv2.waitKey()
cv2.destroyAllWindows()