import cv2

# 读取图像
img = cv2.imread('data/0000.jpg')

# 单个像素取值
red = img[0, 0, 2]
print(red)

# 读取红色分量值
print(img[10, 5, 2])
print(img.item(10, 5, 2))
# 修改红色分量值
img.itemset((10, 5, 2), 100)
print(img.item(10, 5, 2))

# 获取图像中图书区域
book = img[2:1247, 1230:2145]
cv2.imshow("book", book)
cv2.waitKey(0)
print(book.shape)

# 读取蓝色通道，最后一个值为 0
b = book[:, :, 0]
# 读取绿色通道，最后一个值为 1
g = book[:, :, 1]
# 读取红色通道，最后一个值为 2
r = book[:, :, 2]

# 分别显示
cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)

cv2.waitKey()
cv2.destroyAllWindows()

b, g, r = cv2.split(book)        # 图像通道分离
book2 = cv2.merge((b, g, r))     # 图像通道合并
cv2.imshow("book2", book)
cv2.waitKey()
cv2.destroyAllWindows()

# 颜色空间转换
book_gray = cv2.cvtColor(book, cv2.COLOR_BGR2GRAY)
cv2.imshow("book_gray", book_gray)
cv2.waitKey()
cv2.destroyAllWindows()

# 图像边框填充
from matplotlib import pyplot as plt

RED = [255, 0, 0]
img1 = img[1188:1455, 1725:2022]
replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=RED)

plt.subplot(231), plt.axis('off'), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.axis('off'), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.axis('off'), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.axis('off'), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.axis('off'), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.axis('off'), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.show()
