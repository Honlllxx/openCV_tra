import cv2
import numpy as np
import matplotlib.pyplot as plt


# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def main():
    # 读取图像（以灰度模式）
    image_path = "data/img_2.png"  # 可以替换为自己的图像路径
    img = cv2.imread(image_path, 0)

    if img is None:
        print(f"无法读取图像，请检查路径: {image_path}")
        print("将使用默认的 Lena 图像进行演示")
        # 如果无法读取图像，使用OpenCV内置的Lena图像（某些版本可能没有）
        try:
            img = cv2.imread(cv2.samples.findFile("lena.jpg"), 0)
        except:
            # 创建一个简单的测试图像
            img = np.zeros((200, 200), dtype=np.uint8)
            cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
            cv2.circle(img, (100, 100), 30, 0, -1)

    # 1. 生成反向图
    img_inv = 255 - img

    # 2. 阈值分割 - 设置不同参数
    thresh_value = 127  # 阈值
    max_val = 255  # 最大值

    # 不同阈值分割方法
    ret, thresh_binary = cv2.threshold(img, thresh_value, max_val, cv2.THRESH_BINARY)
    ret, thresh_binary_inv = cv2.threshold(img, thresh_value, max_val, cv2.THRESH_BINARY_INV)
    ret, thresh_trunc = cv2.threshold(img, thresh_value, max_val, cv2.THRESH_TRUNC)
    ret, thresh_tozero = cv2.threshold(img, thresh_value, max_val, cv2.THRESH_TOZERO)
    ret, thresh_tozero_inv = cv2.threshold(img, thresh_value, max_val, cv2.THRESH_TOZERO_INV)

    # 3. 形态学操作 - 定义结构元素
    kernel = np.ones((3, 3), np.uint8)  # 3x3的结构元素

    # 膨胀和腐蚀
    img_dilate = cv2.dilate(img, kernel, iterations=1)
    img_erode = cv2.erode(img, kernel, iterations=1)

    # 4. 开运算和闭运算
    img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 显示结果

    # 第一组：原图、反向图和阈值分割(BINARY)
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('原图')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(img_inv, cmap='gray')
    plt.title('反向图')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(thresh_binary, cmap='gray')
    plt.title('阈值分割 (BINARY)')
    plt.axis('off')

    # 第二组：其他阈值分割结果
    plt.subplot(234)
    plt.imshow(thresh_binary_inv, cmap='gray')
    plt.title('阈值分割 (BINARY_INV)')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(thresh_trunc, cmap='gray')
    plt.title('阈值分割 (TRUNC)')
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(thresh_tozero, cmap='gray')
    plt.title('阈值分割 (TOZERO)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 显示TOZERO_INV结果
    plt.figure(figsize=(6, 4))
    plt.imshow(thresh_tozero_inv, cmap='gray')
    plt.title('阈值分割 (TOZERO_INV)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 第三组：膨胀和腐蚀
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('原图')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(img_dilate, cmap='gray')
    plt.title('膨胀结果')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(img_erode, cmap='gray')
    plt.title('腐蚀结果')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 第四组：开运算和闭运算
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('原图')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(img_open, cmap='gray')
    plt.title('开运算结果')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(img_close, cmap='gray')
    plt.title('闭运算结果')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
