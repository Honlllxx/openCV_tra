from skimage.util import random_noise
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("D:\\Test\\1\\514.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("D:\\Test\\2\\514.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.imread("D:\\Test\\3\\500.png")
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.imread("D:\\Test\\4\\514.jpg")
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)



# 显示结果
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 30
plt.rcParams['figure.figsize'] = (19.2, 10.8)
imgs = [img1, img2, img3, img4]
titles = ["原始图像", "人工调整", "训练结果-200epoch", "当前项目上算法"]

for i in range(4):
    img0 = plt.subplot(2, 2, i + 1)
    img0.set_title(titles[i])
    plt.imshow(imgs[i])
    plt.xticks([])
    plt.yticks([])

plt.show()