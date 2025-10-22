import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
img = cv2.imread("data\coins.jpg")
cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("input image", img)

# 均值漂移滤波
blurred = cv2.pyrMeanShiftFiltering(img, 10, 100)
cv2.imshow("blurred image", blurred)


# 二值化处理
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("binary-image", binary)

# 形态学处理，剔除毛刺
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(mb, kernel, iterations=3)
cv2.imshow("mor-opt", sure_bg)

# 距离变换
dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
dist_output = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow("distance-t", dist_output * 50)

# 获取连通区域
ret, surface = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)

surface_fg = np.uint8(surface)
cv2.imshow("surface-bin", surface_fg)
unknown = cv2.subtract(sure_bg, surface_fg)
ret, markers = cv2.connectedComponents(surface_fg)


# 分水岭分割
markers = markers + 1
markers[unknown == 255] = 0
print(type(markers))
print(np.sum(markers == 2))
markers = cv2.watershed(img, markers=markers)
print(np.unique(markers))
print(np.sum(markers == 2))
img[markers == -1] = [0, 0, 255]
cv2.imshow("result", img)
markers = markers + 1
plt.imshow(markers)
plt.show()
cv2.waitKey(0)

