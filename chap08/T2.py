import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像（替换为实际球体图像路径）
image = cv2.imread("data/bga.jpg")
if image is None:
    raise ValueError("无法读取图像，请检查路径是否正确")

# 2. 预处理：转为灰度图并高斯模糊去噪
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯核大小(5,5)，标准差0

# 3. 边缘检测（Canny算子）
edges = cv2.Canny(blurred, 50, 150)  # 低阈值50，高阈值150

# 4. 轮廓提取
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# RETR_EXTERNAL：只保留外轮廓；CHAIN_APPROX_SIMPLE：压缩轮廓点

# 5. 筛选球体轮廓（根据面积特征）
sphere_contours = []
for contour in contours:
    area = cv2.contourArea(contour)  # 计算轮廓面积
    # 假设球体面积在1000~10000之间（需根据实际图像调整阈值）
    if 1000 < area < 10000:
        sphere_contours.append(contour)

# 6. 可视化结果
result = image.copy()
# 在原图上绘制筛选后的轮廓（绿色，线宽2）
cv2.drawContours(result, sphere_contours, -1, (0, 255, 0), 2)

# 显示原图、边缘图、结果图
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("原图")
plt.axis("off")

plt.subplot(132)
plt.imshow(edges, cmap="gray")
plt.title("边缘检测结果")
plt.axis("off")

plt.subplot(133)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("球体轮廓提取结果")
plt.axis("off")

plt.tight_layout()
plt.show()