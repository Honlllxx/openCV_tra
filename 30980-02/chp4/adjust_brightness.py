import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("data/0179.png")

# 亮度因子，加k变亮，减k变暗
[averB, averG, averR] = np.array(cv2.mean(img))[:-1] / 3
k = np.ones((img.shape))
k[:, :, 0] *= averB
k[:, :, 1] *= averG
k[:, :, 2] *= averR

lighten = img + k
lighten[lighten > 255] = 255
lighten[lighten < 0] = 0
lighten = lighten.astype(np.uint8)

# 解决plt 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 30
# 全局设置输出图片大小 1920 x 1080 像素
plt.rcParams['figure.figsize'] = (19.2, 10.8)
imgs = [img, lighten]
titles = ["原始图像", "亮度增强"]

for i in range(2):
    img0 = plt.subplot(1, 2, i + 1)
    img0.set_title(titles[i])
    plt.imshow(imgs[i][:, :, ::-1]) #BGR->RGB
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()