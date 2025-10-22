import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread('data/apple.png', 0)

# 反向
img_inv = 255 - img

# 阈值分割
ret, thresh = cv2.threshold(img_inv, 30, 255, cv2.THRESH_BINARY)

# 显示结果
cv2.imshow("org_img", img)
cv2.imshow("img_inv", img)
cv2.imshow("apple", thresh)
cv2.waitKey()
cv2.destroyAllWindows()

# 解决plt 中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.size'] = 30
# # 全局设置输出图片大小 1920 x 1080 像素
# plt.rcParams['figure.figsize'] = (19.2, 10.8)
# titles = ['原图', '反向', '分割苹果']
# images = [img, img_inv, thresh]
# for i in range(3):
#     plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


img = cv2.imread('data/gradient.png', 0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
plt.rcParams['figure.figsize'] = (19.2, 10.8)
plt.rcParams['font.size'] = 30

titles = ['Original Image', 'BINARY', 'BINARY INV', 'TRUNC', 'TOZERO', 'TOZERO INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 自动阈值分割
img = cv2.imread('data/noisy.png', 0)
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img, (7, 7), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("data/th3.png", th3)
# 绘制图像及直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['噪声图像', '直方图', '阈值分割(v=127)',
          '噪声图像', '直方图', "Otsu分割",
          '高斯模糊', '直方图', "Otsu分割"]
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.show()
