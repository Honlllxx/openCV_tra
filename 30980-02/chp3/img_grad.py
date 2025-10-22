import cv2

img = cv2.imread('data/0499.jpg')
img = cv2.resize(img, dsize=None, fx=0.20, fy=0.20, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 对x方向求导，得到y方向的边缘
rst = cv2.Sobel(img, cv2.CV_64F, 1, 0)
# 对y方向求导，得到x方向的边缘
# rst = cv2.Sobel(img, cv2.CV_64F, 0, 1)
# 对x，y方向同时求导
# rst = cv2.Sobel(img, cv2.CV_64F, 1, 1)
rst = cv2.convertScaleAbs(rst)
cv2.imshow('img', img)
cv2.imshow('rst', rst)
cv2.waitKey()
cv2.destroyAllWindows()

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
cv2.imshow("sobelx", sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
cv2.imshow('sobely', sobely)

# 图像叠加，得到边缘信息
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv2.imshow("sobelxy", sobelxy)

# 反向输出
sobelxy = cv2.subtract(255, sobelxy)
cv2.imshow("sobelxy_invert", sobelxy)

cv2.waitKey()
cv2.destroyAllWindows()

# 边缘提取
laplacian = cv2.Laplacian(img, cv2.CV_64F)
cannyEdge = cv2.Canny(img, 100, 200)
cv2.imshow("laplacian", laplacian)
cv2.imshow("canny", cannyEdge)
cv2.waitKey()
cv2.destroyAllWindows()
