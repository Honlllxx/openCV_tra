from skimage.util import random_noise
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("data/0499.jpg")
img = cv2.resize(img, dsize=None, fx=0.10, fy=0.10, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

snoise_img = random_noise(img, mode='salt', amount=0.05)  # 盐噪声
pnoise_img = random_noise(img, mode='pepper', amount=0.05)  # 椒噪声
spnoise_img = random_noise(img, mode='s&p', amount=0.05)  # 椒盐噪声
guassian_img = random_noise(img, mode='gaussian')  # 高斯噪声
speckle_img = random_noise(img, mode='speckle')  # speckle 均匀噪声/斑点噪声

# 解决plt 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 30
# 全局设置输出图片大小 1920 x 1080 像素
plt.rcParams['figure.figsize'] = (19.2, 10.8)
imgs = [img, snoise_img, pnoise_img, spnoise_img, guassian_img, speckle_img]
titles = ["原始图像", "盐噪声", "椒噪声", "椒盐噪声", "高斯噪声", "斑点噪声"]

for i in range(6):
    img0 = plt.subplot(2, 3, i + 1)
    img0.set_title(titles[i])
    plt.imshow(imgs[i], cmap="gray")
    plt.xticks([])
    plt.yticks([])

plt.show()
cv2.waitKey(0)


