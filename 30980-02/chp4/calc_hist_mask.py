import numpy as np
import cv2
import matplotlib.pyplot as plt

# 直接以灰度图的方式读入
img = cv2.imread("data/0678.png", 0)

# 创建蒙版
mask = np.zeros(img.shape[:2], np.uint8)
mask[200:600, 400:1000] = 1

# 掩膜
mask_img = cv2.bitwise_and(img, img, mask=mask)

# 统计掩膜后图像的灰度图
mask_histr = cv2.calcHist([img], [0], mask, [256], [0, 256])

# 解决plt 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 30
# 全局设置输出图片大小 1920 x 1080 像素
plt.rcParams['figure.figsize'] = (19.2, 10.8)
imgs = [img, mask, mask_img]
titles = ["原始图像", "掩膜", "掩膜图像", "直方图"]

for i in range(3):
    img0 = plt.subplot(2, 2, i + 1)
    img0.set_title(titles[i])
    plt.imshow(imgs[i], cmap="gray")
    plt.xticks([])
    plt.yticks([])
img4 = plt.subplot(2, 2, 4)
img4.set_title(titles[3])
plt.plot(mask_histr)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
