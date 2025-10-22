import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("data/0678.png")

# 对比度因子，<1 降低对比度 >1 增加对比度
coefficent = 1.5
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
m = cv2.mean(img)[0]
graynew = m + coefficent * (imggray - m)
contrast = np.zeros(img.shape, np.float32)
k = np.divide(graynew, imggray, out=np.zeros_like(graynew), where=imggray != 0)
contrast[:, :, 0] = img[:, :, 0] * k
contrast[:, :, 1] = img[:, :, 1] * k
contrast[:, :, 2] = img[:, :, 2] * k
contrast[contrast > 255] = 255
contrast[contrast < 0] = 0

contrast = contrast.astype(np.uint8)

# 解决plt 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 30
# 全局设置输出图片大小 1920 x 1080 像素
plt.rcParams['figure.figsize'] = (19.2, 10.8)
imgs = [img, contrast]
titles = ["原始图像", "对比度调节"]

for i in range(2):
    img0 = plt.subplot(1, 2, i + 1)
    img0.set_title(titles[i])
    plt.imshow(imgs[i][:, :, ::-1]) #BGR->RGB
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()