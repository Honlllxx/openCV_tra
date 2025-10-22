import cv2
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("data/0179.png")
histr = cv2.calcHist([img], [0], None, [255], [0, 256])
# 显示图像
plt.imshow(img[:, :, :: -1])
plt.show()

# 显示图像直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('灰度值')
plt.ylabel('等于该灰度值的像素个数')
plt.plot(histr)
plt.grid()
plt.show()

# 以灰度图形式读取图像
img = cv2.imread("data/0179.png", 0)

# 直方图均衡化处理
# dst = cv.equalizeHist(img)
# img: 灰度图像
# dist: 均衡化后的结果
dst = cv2.equalizeHist(img)
cv2.imwrite("data/equalize_dst.png", dst)

# 显示图像
plt.imshow(img, cmap=plt.cm.gray)
plt.show()
plt.imshow(dst, cmap=plt.cm.gray)
plt.show()
histr = cv2.calcHist([dst], [0], None, [255], [0, 256])
# 显示图像直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('灰度值')
plt.ylabel('等于该灰度值的像素个数')
plt.plot(histr)
plt.show()

# 自适应直方图均衡化
# 创建一个自适应均衡化对象，并应用于图像
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
cv2.imwrite("data/clahe_dst.png", cl1)
# 显示图像
plt.imshow(cl1, cmap=plt.cm.gray)
plt.show()

histr = cv2.calcHist([cl1], [0], None, [255], [0, 256])
# 显示图像直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('灰度值')
plt.ylabel('等于该灰度值的像素个数')
plt.plot(histr)
plt.show()