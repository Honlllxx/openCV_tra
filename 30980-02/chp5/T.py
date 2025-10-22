import cv2
import numpy as np
from matplotlib import pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取图片
img = cv2.imread("data\coins.jpg")
if img is None:
    raise ValueError("无法读取图片，请检查文件路径是否正确")

# 转换为RGB格式用于matplotlib显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 均值漂移滤波
blurred = cv2.pyrMeanShiftFiltering(img, 10, 100)
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

# 二值化处理
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 形态学处理，剔除毛刺
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(mb, kernel, iterations=3)

# 距离变换
dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
dist_output = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)

# 获取连通区域
ret, surface = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)
surface_fg = np.uint8(surface)
unknown = cv2.subtract(sure_bg, surface_fg)
ret, markers = cv2.connectedComponents(surface_fg)

# 分水岭分割
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers=markers)
img[markers == -1] = [0, 0, 255]  # 边界标记为红色
result_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
markers_plot = markers + 1  # 调整标记值用于显示

# 创建子图显示所有结果
plt.figure(figsize=(15, 12))

# 原图
plt.subplot(3, 3, 1)
plt.imshow(img_rgb)
plt.title("原始图像")
plt.axis('off')

# 均值漂移滤波结果
plt.subplot(3, 3, 2)
plt.imshow(blurred_rgb)
plt.title("均值漂移滤波")
plt.axis('off')

# 灰度图
plt.subplot(3, 3, 3)
plt.imshow(gray, cmap='gray')
plt.title("灰度图像")
plt.axis('off')

# 二值化结果
plt.subplot(3, 3, 4)
plt.imshow(binary, cmap='gray')
plt.title("二值化图像")
plt.axis('off')

# 形态学处理结果
plt.subplot(3, 3, 5)
plt.imshow(sure_bg, cmap='gray')
plt.title("形态学处理")
plt.axis('off')

# 距离变换结果
plt.subplot(3, 3, 6)
plt.imshow(dist_output, cmap='gray')
plt.title("距离变换")
plt.axis('off')

# 前景提取结果
plt.subplot(3, 3, 7)
plt.imshow(surface_fg, cmap='gray')
plt.title("前景提取")
plt.axis('off')

# 分水岭分割结果
plt.subplot(3, 3, 8)
plt.imshow(result_rgb)
plt.title("分水岭分割结果")
plt.axis('off')

# 标记图像
plt.subplot(3, 3, 9)
plt.imshow(markers_plot, cmap='tab20')
plt.title("标记图像")
plt.axis('off')

plt.tight_layout()  # 调整布局
plt.show()