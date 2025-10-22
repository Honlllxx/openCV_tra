from skimage.util import random_noise
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("data/0499.jpg")
img = cv2.resize(img, dsize=None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 椒盐噪声
pepper_noise = random_noise(img, mode='pepper')  # 高斯噪声

# 中值滤波
img_medianBlur = cv2.medianBlur(img, 3)

# 显示结果
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 30
plt.rcParams['figure.figsize'] = (19.2, 10.8)
imgs = [img, pepper_noise, img_medianBlur]
titles = ["原始图像", "椒盐噪声", "中值滤波"]

for i in range(3):
    img0 = plt.subplot(1, 3, i + 1)
    img0.set_title(titles[i])
    plt.imshow(imgs[i], cmap="gray")
    plt.xticks([])
    plt.yticks([])

plt.show()



