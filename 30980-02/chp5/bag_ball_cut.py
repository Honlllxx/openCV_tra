import cv2
import numpy as np

def cvshow(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 720, 720)
    cv2.imshow(name, img)

# 读取图片
img = cv2.imread('data/bga.jpg', 0)
cvshow('bga', img)

# 阈值分割，得到二值图像
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cvshow('binary', thresh)

# 获取图像外部轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 显示图像轮廓，供下一步分析
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_bgr, contours, -1, (0, 0, 255), 3)
cvshow('contours', img_bgr)

balls = list()
# 过滤球轮廓
for contour in contours:
    # 计算锡球面积
    ballArea = cv2.contourArea(contour)
    center, radius = cv2.minEnclosingCircle(contour)
    circleArea = np.pi * radius * radius
    roundness = ballArea / circleArea
    # 根据圆度及半径过滤，更多的过滤条件可以结合工程需求
    if roundness > 0.83  and 9 <= radius <= 25:
       balls.append(contour)

img_ball = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_ball, balls, -1, (0, 0, 255), 3)
cvshow('balls', img_ball)
cv2.waitKey(0)
cv2.destroyAllWindows()