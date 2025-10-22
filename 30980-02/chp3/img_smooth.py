
import cv2
import matplotlib.pyplot as plt

# 输入图像
img = cv2.imread('data/0499.jpg')
img = cv2.resize(img, dsize=None, fx=0.10, fy=0.10, interpolation=cv2.INTER_CUBIC)

# 均值滤波
img_blur = cv2.blur(img, (3,3)) # (3,3)代表卷积核尺寸，随着尺寸变大，图像会越来越模糊
img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB) # BGR转化为RGB格式

# 方框滤波
img_boxFilter1 = cv2.boxFilter(img, -1, (3,3), normalize=True) # 当 normalize=True 时，与均值滤波结果相同
img_boxFilter1 = cv2.cvtColor(img_boxFilter1, cv2.COLOR_BGR2RGB) # BGR转化为RGB格式

img_boxFilter2 = cv2.boxFilter(img, -1, (3,3), normalize=False)
img_boxFilter2 = cv2.cvtColor(img_boxFilter2, cv2.COLOR_BGR2RGB) # BGR转化为RGB格式

# 高斯滤波
img_GaussianBlur= cv2.GaussianBlur(img, (3,3), 0, 0) # 参数说明：(源图像，核大小，x方向的标准差，y方向的标准差)
img_GaussianBlur = cv2.cvtColor(img_GaussianBlur, cv2.COLOR_BGR2RGB) # BGR转化为RGB格式

# 中值滤波
img_medianBlur = cv2.medianBlur(img, 3)
img_medianBlur = cv2.cvtColor(img_medianBlur, cv2.COLOR_BGR2RGB) # BGR转化为RGB格式

# 双边滤波
# 参数说明：(源图像，核大小，sigmaColor，sigmaSpace)
img_bilateralFilter=cv2.bilateralFilter(img, 50, 100, 100)
img_bilateralFilter = cv2.cvtColor(img_bilateralFilter, cv2.COLOR_BGR2RGB) # BGR转化为RGB格式

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 30
# 全局设置输出图片大小 1920 x 1080 像素
plt.rcParams['figure.figsize'] = (19.2, 10.8)
titles = ['均值平滑', '方框滤波', '方框滤波2',
          '高斯平滑', '中值平滑', '双边滤波']
images = [img_blur, img_boxFilter1, img_boxFilter2, img_GaussianBlur, img_medianBlur, img_bilateralFilter]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i]), plt.title(titles[i])
    plt.axis('off')
plt.show()