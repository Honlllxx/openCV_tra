import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# 低通滤波
def Low_Pass_Filter(srcImg_path):
    # img = cv.imread('srcImg_path', 0)
    img = np.array(Image.open(srcImg_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置低通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    return res


# 高通滤波
def High_Pass_Filter(srcImg_path):
    # img = cv.imread(srcImg_path, 0)
    img = np.array(Image.open(srcImg_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置高通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    return res

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 30
# 全局设置输出图片大小 1920 x 1080 像素
plt.rcParams['figure.figsize'] = (19.2, 10.8)
img_Low_Pass_Filter = Low_Pass_Filter('data/0499.jpg')
plt.subplot(121), plt.imshow(img_Low_Pass_Filter, 'gray'), plt.title('低通滤波')
plt.axis('off')

img_High_Pass_Filter = High_Pass_Filter('data/0499.jpg')
plt.subplot(122), plt.imshow(img_High_Pass_Filter, 'gray'), plt.title('高通滤波')
plt.axis('off')

plt.show()
