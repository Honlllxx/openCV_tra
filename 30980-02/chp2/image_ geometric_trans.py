import cv2

img = cv2.imread('data/0000.jpg')
# 获取图像宽度和高度, 注意OpenCV矩阵索引第1维为行（高度），第2维为列（宽度）
height, width = img.shape[:2]

# 图像缩放四分之一大小
res1 = cv2.resize(img, (int(width / 4), int(height / 4)), interpolation=cv2.INTER_CUBIC)
cv2.imshow('resize1', res1)

# 第二种缩放方式，fx为横轴缩放比例，fy为纵轴缩放比例
res2 = cv2.resize(img, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

cv2.imshow('resize2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 导入矩阵库numpy
import numpy as np

rows, cols = res1.shape[:2]
# 构建变换矩阵
M = np.float32([[1, 0, 100], [0, 1, 50]])
# 位移操作
translate = cv2.warpAffine(res1, M, (cols, rows))

cv2.imshow('translate', translate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 构建旋转矩阵，这里cols-1和rows-1 是图像坐标值上限
M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1)
rotate = cv2.warpAffine(res1, M, (cols, rows))

cv2.imshow('rotate', rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存储res1。jpg, 方便进行标定
cv2.imwrite("data/res1.jpg", res1)

# 透视变换对应关系
pts1 = np.float32([[69, 0], [563, 8], [25, 479], [601, 479]])
pts2 = np.float32([[0, 0], [576, 0], [0, 479], [576, 479]])
M = cv2.getPerspectiveTransform(pts1, pts2)
warp_persp = cv2.warpPerspective(res1, M, (576, 479))

# 存储变形后图像
cv2.imwrite("data/warp_persp.jpg", warp_persp)
# 显示图像
cv2.imshow('perspective', warp_persp)
cv2.waitKey(0)
cv2.destroyAllWindows()
