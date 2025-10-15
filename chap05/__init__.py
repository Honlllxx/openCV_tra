import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager  # 用于设置中文字体


def adjust_brightness(image, value):
    """调整图像亮度（修复uint8溢出和类型转换问题）"""
    if len(image.shape) == 3:  # 彩色图像：转HSV调整亮度（避免颜色失真）
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 关键修复：用cv2.add避免溢出（自动截断0-255范围）
        v = cv2.add(v, value)  # 替代原v[v <= lim] += value，无需手动判断范围

        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    else:  # 灰度图像：直接调整
        return cv2.add(image, value)


def show_histogram(image, title, ax):
    """显示图像的直方图"""
    if len(image.shape) == 3:  # 彩色图像（BGR转RGB后显示）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors = ('r', 'g', 'b')  # Matplotlib显示RGB通道，与OpenCV的BGR对应
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
            ax.plot(hist, color=col, label=f'{col.upper()}通道')
        ax.legend()
    else:  # 灰度图像
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='gray', label='灰度通道')
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('像素值（0-255）')
    ax.set_ylabel('像素数量')
    ax.set_xlim([0, 256])


def adjust_contrast(image, alpha):
    """调整图像对比度（alpha>1增强，0<alpha<1减弱）"""
    # cv2.convertScaleAbs自动处理溢出，beta=0表示不调整亮度
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def histogram_mask_operation(image):
    """直方图掩膜操作（创建圆形掩膜，提取感兴趣区域）"""
    # 确保输入为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 创建圆形掩膜（与图像同尺寸）
    mask = np.zeros_like(gray)
    rows, cols = gray.shape
    center = (cols // 2, rows // 2)
    radius = min(center) // 2  # 圆的半径为图像最小半长的1/2
    cv2.circle(mask, center, radius, 255, -1)  # -1表示填充圆

    # 应用掩膜（仅保留圆内区域）
    masked_image = cv2.bitwise_and(gray, gray, mask=mask)

    # 计算掩膜区域的直方图（仅统计圆内像素）
    hist_mask = cv2.calcHist([gray], [0], mask, [256], [0, 256])

    return gray, mask, masked_image, hist_mask


# -------------------------- 主函数（流程优化，增加错误处理） --------------------------
def main():
    # 1. 读取图像（支持用户输入路径，增加错误处理）
    image_path = input("请输入图像路径（例如：test.jpg，默认为lena.jpg）: ").strip()
    if not image_path:
        image_path = "lena.jpg"  # 默认图像（需确保该图像在代码同级目录）

    # 读取图像并处理异常
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取图像 {image_path}，使用默认测试图像！")
        # 创建测试图像（黑底+白色矩形+灰色圆形）
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (350, 350), (255, 255, 255), -1)
        cv2.circle(image, (200, 200), 100, (128, 128, 128), -1)

    # 转换为RGB用于Matplotlib显示（OpenCV默认BGR）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------- 1. 显示图像及其直方图 --------------------------
    plt.figure(figsize=(14, 6), dpi=100)  # dpi提高图像清晰度

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('原始彩色图像')
    plt.axis('off')  # 隐藏坐标轴

    plt.subplot(1, 2, 2)
    show_histogram(image, '原始图像直方图', plt.gca())

    plt.tight_layout()
    plt.show()

    # -------------------------- 2. 直方图均衡化 vs CLAHE --------------------------
    # 普通直方图均衡化（全局均衡，可能过曝）
    equalized_global = cv2.equalizeHist(gray_image)
    # CLAHE（限制对比度自适应均衡化，避免局部过曝）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # clipLimit控制对比度
    equalized_clahe = clahe.apply(gray_image)

    plt.figure(figsize=(16, 8), dpi=100)

    # 原始灰度图
    plt.subplot(2, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('原始灰度图像')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    show_histogram(gray_image, '原始灰度直方图', plt.gca())

    # 全局均衡化
    plt.subplot(2, 3, 2)
    plt.imshow(equalized_global, cmap='gray')
    plt.title('全局直方图均衡化')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    show_histogram(equalized_global, '全局均衡化直方图', plt.gca())

    # CLAHE均衡化
    plt.subplot(2, 3, 3)
    plt.imshow(equalized_clahe, cmap='gray')
    plt.title('CLAHE自适应均衡化')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    show_histogram(equalized_clahe, 'CLAHE均衡化直方图', plt.gca())

    plt.tight_layout()
    plt.show()

    # -------------------------- 3. 直方图掩膜操作 --------------------------
    gray, mask, masked_image, hist_mask = histogram_mask_operation(image)

    plt.figure(figsize=(16, 8), dpi=100)

    plt.subplot(1, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('原始灰度图像')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('圆形掩膜（白色为有效区域）')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(masked_image, cmap='gray')
    plt.title('掩膜后的图像（仅保留圆内）')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.plot(hist_mask, color='gray')
    plt.title('掩膜区域直方图')
    plt.xlabel('像素值（0-255）')
    plt.ylabel('像素数量')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

    # -------------------------- 4. 调整图像亮度 --------------------------
    brightness_values = [50, -50]  # 亮度+50（变亮）、亮度-50（变暗）
    bright_images = [adjust_brightness(image, val) for val in brightness_values]
    bright_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bright_images]

    plt.figure(figsize=(16, 8), dpi=100)

    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title('原始图像（亮度=0）')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    show_histogram(image, '原始图像直方图', plt.gca())

    # 亮度增加50
    plt.subplot(2, 2, 3)
    plt.imshow(bright_images_rgb[0])
    plt.title(f'亮度增加 {brightness_values[0]}（更亮）')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    show_histogram(bright_images[0], f'亮度+{brightness_values[0]} 直方图', plt.gca())

    plt.tight_layout()
    plt.show()

    # 亮度减少50（单独显示，避免拥挤）
    plt.figure(figsize=(16, 4), dpi=100)

    plt.subplot(1, 2, 1)
    plt.imshow(bright_images_rgb[1])
    plt.title(f'亮度减少 {abs(brightness_values[1])}（更暗）')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    show_histogram(bright_images[1], f'亮度-{abs(brightness_values[1])} 直方图', plt.gca())

    plt.tight_layout()
    plt.show()

    # -------------------------- 5. 调整图像对比度 --------------------------
    contrast_values = [1.8, 0.5]  # 对比度+80%（alpha=1.8）、对比度-50%（alpha=0.5）
    contrast_images = [adjust_contrast(image, val) for val in contrast_values]
    contrast_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in contrast_images]

    plt.figure(figsize=(16, 8), dpi=100)

    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title('原始图像（对比度=1.0）')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    show_histogram(image, '原始图像直方图', plt.gca())

    # 对比度增加（alpha=1.8）
    plt.subplot(2, 2, 3)
    plt.imshow(contrast_images_rgb[0])
    plt.title(f'对比度增强（α={contrast_values[0]}，明暗差异更大）')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    show_histogram(contrast_images[0], f'对比度α={contrast_values[0]} 直方图', plt.gca())

    plt.tight_layout()
    plt.show()

    # 对比度减少（alpha=0.5）
    plt.figure(figsize=(16, 4), dpi=100)

    plt.subplot(1, 2, 1)
    plt.imshow(contrast_images_rgb[1])
    plt.title(f'对比度减弱（α={contrast_values[1]}，明暗差异更小）')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    show_histogram(contrast_images[1], f'对比度α={contrast_values[1]} 直方图', plt.gca())

    plt.tight_layout()
    plt.show()

    print("所有图像处理任务完成！")


if __name__ == "__main__":
    main()