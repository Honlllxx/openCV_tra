import cv2
from paddleocr import PaddleOCR
from matplotlib import pyplot as plt

img = cv2.imread("data/car/025-89_92-331&290_585&392-592&383_347&391_348&293_593&285-0_0_15_33_10_32_27-87-60.jpg")

# 1、转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

fig = plt.figure(figsize=(6, 6))
plt.imshow(gray, "gray"), plt.axis('off'), plt.title("gray")
plt.show()

# 2、顶帽运算
# 创建一个17*17矩阵内核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
# cv2.morphologyEx：形态学操作，将腐蚀膨胀结合使用
# cv2.MORPH_TOPHAT ：顶帽操作（原图像-开运算结果：突出灰度中亮的区域）
#tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

fig = plt.figure(figsize=(6, 6))
plt.imshow(tophat, "gray"), plt.axis('off'), plt.title("tophat")
plt.show()

# 3、Sobel算子提取y方向边缘（揉成一坨）
y = cv2.Sobel(tophat, cv2.CV_16S, 1, 0)
absY = cv2.convertScaleAbs(y)

fig = plt.figure(figsize=(6, 6))
plt.imshow(absY, "gray"), plt.axis('off'), plt.title("sobelY")
plt.show()

# 4、自适应二值化（阈值自己可调）
ret, binary = cv2.threshold(absY, 75, 255, cv2.THRESH_BINARY)

fig = plt.figure(figsize=(6, 6))
plt.imshow(binary, "gray"), plt.axis('off'), plt.title("binary")
plt.show()

# 5、开运算分割（纵向去噪，分隔）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
Open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

fig = plt.figure(figsize=(6, 6))
plt.imshow(Open, "gray"), plt.axis('off'), plt.title("Open")
plt.show()

# 6、闭运算合并，把图像闭合、揉团，使图像区域化，便于找到车牌区域，进而得到轮廓
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 15))
close = cv2.morphologyEx(Open, cv2.MORPH_CLOSE, kernel)

fig = plt.figure(figsize=(6, 6))
plt.imshow(close, "gray"), plt.axis('off'), plt.title("close")
plt.show()

# 7、膨胀/腐蚀（去噪得到车牌区域）
# 中远距离车牌识别
kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
# 近距离车牌识别
# kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (79, 15))
# kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
# 7-1、腐蚀、膨胀（去噪）
erode_y = cv2.morphologyEx(close, cv2.MORPH_ERODE, kernel_y)
dilate_y = cv2.morphologyEx(erode_y, cv2.MORPH_DILATE, kernel_y)

# 7-2、膨胀、腐蚀（连接）（二次缝合）
dilate_x = cv2.morphologyEx(dilate_y, cv2.MORPH_DILATE, kernel_x)
erode_x = cv2.morphologyEx(dilate_x, cv2.MORPH_ERODE, kernel_x)

fig = plt.figure(figsize=(15, 15))
plt.subplot(131), plt.imshow(erode_y, "gray"), plt.axis('off'), plt.title("erode_y")
plt.subplot(132), plt.imshow(dilate_y, 'gray'), plt.axis('off'), plt.title("dilate_y")
plt.subplot(133), plt.imshow(erode_x, 'gray'), plt.axis('off'), plt.title("erode_x")
plt.show()

# 8、腐蚀、膨胀：去噪
kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
erode = cv2.morphologyEx(erode_x, cv2.MORPH_ERODE, kernel_e)

kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 11))
dilate = cv2.morphologyEx(erode, cv2.MORPH_DILATE, kernel_d)

fig = plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(erode, "gray"), plt.axis('off'), plt.title("erode")
plt.subplot(122), plt.imshow(dilate, 'gray'), plt.axis('off'), plt.title("dilate")
plt.show()

# 9、获取外轮廓
img_copy = img.copy()

# 9-1、得到轮廓
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 9-2、画出轮廓并显示
cv2.drawContours(img_copy, contours, -1, (255, 0, 255), 2)

fig = plt.figure(figsize=(6, 6))
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.imshow(img_copy, "gray"), plt.axis('off'), plt.title("Contours")
plt.show()

#10、遍历所有轮廓，找到车牌轮廓
count = 0

for contour in contours:
    area = cv2.contourArea(contour)  # 计算轮廓内区域的面积
    # 10-1、得到矩形区域：左顶点坐标、宽和高
    x, y, w, h = cv2.boundingRect(contour)  # 获取坐标值和宽度、高度

    # 10-2、获取轮廓区域的形状信息
    perimeter = cv2.arcLength(contour, True)  # 计算轮廓周长
    # 以精度0.02近似拟合轮廓
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)  # 获取轮廓角点坐标
    CornerNum = len(approx)  # 轮廓角点的数量
    # cv2.polylines(img_copy1, [approx], True, (0, 255, 0), 3)
    # cv2.imshow('approx',  img_copy1)
    # cv2.waitKey(0)

    # 10-2、判断宽高比例、面积、轮廓角点数量，截取符合图片
    # if (w > h * 3 and w < h * 7 )and area>1000 and CornerNum<=5:
    if h * 3 < w < h * 7 and area > 1000:
        # print(count)
        # print(f"CornerNum：{CornerNum}，area：{area}")
        # 截取车牌并显示
        # print(x, y, w, h)
        img = img[(y - 5):(y + h + 5), (x - 5):(x + w + 5)]  # 高，宽
        try:
            count += 1

            fig = plt.figure(figsize=(6, 6))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img), plt.axis('off'), plt.title("img")
            plt.show()

        except:
            img = None
            print("ROI提取出错！")


# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
 # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=False, use_gpu=False,
                    lang="ch", show_log=False)  # need to run only once to download and load model into memory



if img is None:
    print("没有提取到车牌")
    exit()

ocr_text = ocr.ocr(img, cls=False)
for line in ocr_text:
    number_plate = line[-1][-1][0]
    print("车牌：", end="")
    print(number_plate)