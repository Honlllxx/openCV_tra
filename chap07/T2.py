import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_contours(image_path):
    """提取图像轮廓并可视化（进一步优化版）"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊增强，核尺寸调整为7x7
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # 自适应阈值参数微调
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )

    # 形态学操作核尺寸调整为5x5，迭代次数优化
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

    # 提取轮廓，选择RETR_EXTERNAL确保只取最外层轮廓（硬币为独立物体）
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制不同颜色轮廓，增强区分度
    image_with_contours = image.copy()
    for i, cnt in enumerate(contours):
        color = np.random.randint(0, 255, 3).tolist()
        cv2.drawContours(image_with_contours, [cnt], -1, color, 2)

    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("原图")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("优化后轮廓提取结果（多色区分）")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return image, gray, contours


def watershed_segmentation(image, gray):
    """分水岭分割（精准区分版）"""
    # 预处理优化
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学操作核与迭代次数调整
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    sure_bg = cv2.dilate(opening, kernel, iterations=5)

    # 距离变换与阈值优化
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # 换回L2距离，提升精度
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)  # 阈值降至0.3倍
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 连通区域标记
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 应用分水岭
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 255]  # 调整为亮青色，增强视觉区分

    # 彩色类别标记图优化
    colors = np.random.randint(0, 255, size=(ret + 2, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    colors[1] = [30, 30, 30]
    markers_color = colors[markers]

    # 显示结果
    plt.figure(figsize=(20, 8))
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("原图")
    plt.axis("off")

    plt.subplot(142)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("分水岭分割结果（亮青色边界）")
    plt.axis("off")

    plt.subplot(143)
    plt.imshow(cv2.cvtColor(markers_color, cv2.COLOR_BGR2RGB))
    plt.title("彩色分割类别标记图")
    plt.axis("off")

    plt.subplot(144)
    plt.imshow(dist_transform, cmap="gray")
    plt.title("距离变换图（亮处为确定前景）")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "./data/coins.jpg"  # 替换为你的图片路径
    image, gray, _ = extract_contours(image_path)
    watershed_segmentation(image, gray)