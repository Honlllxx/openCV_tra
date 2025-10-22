import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def extract_contours(image_path):
    """提取图像轮廓并在原图上显示"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None, None

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊去除噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上绘制轮廓
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    print(f"找到 {len(contours)} 个轮廓")

    return image, image_with_contours


def watershed_segmentation(image_path):
    """使用分水岭算法进行图像分割"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None, None, None

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 阈值处理，得到二值图像
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去除噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 寻找前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记连通区域
    ret, markers = cv2.connectedComponents(sure_fg)

    # 为所有标记加1，确保背景不是0，而是1
    markers = markers + 1

    # 未知区域标记为0
    markers[unknown == 255] = 0

    # 应用分水岭算法
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # 边界标记为红色

    # 准备分割类别标记图
    markers_normalized = ((markers - markers.min()) / (markers.max() - markers.min()) * 255).astype(np.uint8)
    markers_colored = cv2.applyColorMap(markers_normalized, cv2.COLORMAP_JET)

    return image, markers_colored, markers


def main(image_path):
    """主函数：执行轮廓提取和分水岭分割并显示结果"""
    # 轮廓提取
    original_img, contour_img = extract_contours(image_path)
    if original_img is None:
        return

    # 分水岭分割
    segmented_img, markers_colored, markers = watershed_segmentation(image_path)
    if segmented_img is None:
        return

    # 显示结果
    plt.figure(figsize=(15, 12))

    # 显示原图和轮廓图
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('原图')
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title('轮廓提取结果')
    plt.axis('off')

    # 显示分水岭分割结果
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    plt.title('分水岭分割结果')
    plt.axis('off')

    # 显示分割类别标记图
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(markers_colored, cv2.COLOR_BGR2RGB))
    plt.title('分割类别标记图')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 替换为你的图像路径
    image_path = "./data/coins.jpg"  # 请在此处输入你的图像路径
    main(image_path)