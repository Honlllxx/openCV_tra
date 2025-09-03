import cv2
import numpy as np

# 读取图像
img = cv2.imread('img/image_read.jpg')

if img is not None:
    print("图像读取成功，尺寸:", img.shape)
    # 显示图像
    cv2.imshow('Test Image', img)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()
else:
    print("无法读取图像")