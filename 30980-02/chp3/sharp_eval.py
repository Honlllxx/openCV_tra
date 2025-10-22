import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob

# 读取图片
circle = cv.imread("data/circle.png")
cv.imwrite("data/circle00.png", circle)

# 高斯模糊，模拟镜头对焦
blurrange = range(0, 55, 5)
for i in blurrange:
    if i == 0:
        continue
    blur = cv.GaussianBlur(cv.imread("data/circle00.png"), (0, 0), i)
    cv.imwrite(f"data/circle{i:02d}.png", blur)

# 显示当前结果
images = sorted(glob.glob("data/*[0-9].png"))

fig, axes = plt.subplots(2,len(images)//2)
for i,ax in enumerate(axes.flatten()):
    ax.imshow(cv.imread(images[i]))
    ax.annotate(f"SigmaX = {i*5}", (0,0))
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# 清晰度评价函数
def tenengrad(img, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    Gx = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx*Gx + Gy*Gy
    mn = cv.mean(FM)[0]
    if np.isnan(mn):
        return np.nanmean(FM)
    return mn

# 评估图片
tenengrads = np.empty(len(images))
for i,file in enumerate(sorted(glob.glob("data/*[0-9].png"))):
    tenengrads[i] = tenengrad(cv.imread(file))
print(tenengrads)

# 可视化
fig, ax = plt.subplots()

ax.plot(blurrange[1:], tenengrads[1:], ".")
ax.set_xlabel("Gaußian blur sigmaX")
ax.set_ylabel("Tenengrad")
plt.tight_layout()
plt.show()