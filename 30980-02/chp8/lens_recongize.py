import cv2
from paddleocr import PaddleOCR
import numpy as np
import math

img = cv2.imread('data/lens.jpeg')
img_copy = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)

# 查找图中的圆形轮廓
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, \
                           param1=200, param2=30, minRadius=260, maxRadius=300)

isNG = False
if circles is None:
    print("找圆失败！")

else:
    circles = np.uint16(np.around(circles))
    a, b, c = circles.shape
    print(circles.shape)
    for i in range(b):
        cv2.circle(img_copy, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(img_copy, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3,
                   cv2.LINE_AA)  # draw center of circle
    cv2.imshow("findCircle", img_copy)

# 基于找到的圆做极坐标变换，将文字转换到水平方向
x = circles[0][i][0] - circles[0][i][2]
y = circles[0][i][1] - circles[0][i][2]
w = h = 2 * circles[0][i][2]
center = (circles[0][i][0], circles[0][i][1])
radius = circles[0][i][2]
C = 2 * math.pi * radius
print(C, radius)

ROI = img[y:y + h, x:x + w].copy()
cv2.imshow('ROI', ROI)
trans_center = (center[0] - x, center[1] - y)
polarImg = cv2.warpPolar(ROI, (int(radius), int(C)), trans_center, radius, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
polarImg = cv2.flip(polarImg, 1)  # 镜像
polarImg = cv2.transpose(polarImg)  # 转置
cv2.imshow('polarImg', polarImg)

# 文字识别
ocr = PaddleOCR(use_angle_cls=False, use_gpu=False,
                lang="ch", show_log=False)  # need to run only once to download and load model into memory

result = ocr.ocr(polarImg, cls=False)

for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print('----------------------------')
        print(line)
        pt1 = ((int)(line[0][0][0]), (int)(line[0][0][1]))
        pt2 = ((int)(line[0][1][0]), (int)(line[0][1][1]))
        pt3 = ((int)(line[0][2][0]), (int)(line[0][2][1]))
        pt4 = ((int)(line[0][3][0]), (int)(line[0][3][1]))
        cv2.line(polarImg, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(polarImg, pt2, pt3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(polarImg, pt3, pt4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(polarImg, pt1, pt4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(polarImg, line[1][0], (pt1[0], pt1[1] + 40), 0, 0.6, (0, 0, 255), 1)

cv2.imshow('polarImg-OCR', polarImg)

# 极坐标反变换，将包含识别结果的图像还原成圆形
polarImg = cv2.flip(polarImg, 0)  # 镜像
polarImg = cv2.transpose(polarImg)  # 转置
polarImg_Inv = cv2.warpPolar(polarImg, (w, h), trans_center, radius, cv2.INTER_LINEAR + \
                             cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
cv2.imshow('polarImg_Inv', polarImg_Inv)
cv2.waitKey(0)

# 创建圆形mask图像做copyTo操作，使圆形外部图像与原图保持一致
mask = np.zeros((h,w,1),np.uint8)
cv2.circle(mask,trans_center,radius-3,(255,255,255),-1, cv2.LINE_AA)
cv2.imshow('mask', mask)
ROI = img[y:y+h,x:x+w]
for i in range(0,ROI.shape[0]):
    for j in range(0, ROI.shape[1]):
      if mask[i,j] > 0:
        ROI[i,j] = polarImg_Inv[i,j]
cv2.imshow('result', img)

cv2.waitKey(0)
