import cv2
import numpy as np

# 读取摄像头
cap = cv2.VideoCapture(0)
while (1):
    # 获取图像帧
    _, frame = cap.read()
    # 颜色空间转换： BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 在HSV空间中定义追踪的颜色阈值范围
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # 获取符合阈值的图像掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 通过掩码运算来从原图像上获取目标区域
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
