import cv2
import numpy as np
import matplotlib.pyplot as plt


def cvshow(name, img, figsize=(8, 8)):
    # 转换BGR到RGB（如果是彩色图），确保显示颜色正确
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.title(name)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def detect_balls(image_path, min_radius=9, max_radius=25, circularity_thresh=0.83):
    #读取图像（灰度图）
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    cvshow("原始灰度图", img)

    #  预处理：高斯模糊去噪（减少干扰轮廓）
    blur_kernel = (5, 5)  # 高斯核大小，可根据噪声程度调整
    img_blur = cv2.GaussianBlur(img, blur_kernel, 0)
    cvshow("高斯模糊去噪后", img_blur)

    # 自适应阈值分割（比OTSU更鲁棒，适应光照不均）
    # 使用自适应阈值处理，避免单一阈值导致的过分割/欠分割
    thresh = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯加权计算阈值
        cv2.THRESH_BINARY_INV,  # 反转阈值（目标为白色）
        11,  # 块大小
        2  # 常数项，从均值中减去
    )
    cvshow("自适应阈值分割", thresh)

    # 提取外部轮廓（只保留最外层，减少内部细节干扰）
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,  # 只提取外轮廓
        cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
    )

    # 显示所有轮廓（用于调试）
    img_all_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_all_contours, contours, -1, (0, 0, 255), 2)
    cvshow(f"检测到的所有轮廓（共{len(contours)}个）", img_all_contours)

    # 5. 筛选球体轮廓（基于圆形度和大小）
    balls = []
    for cnt in contours:
        # 计算轮廓面积（避免过小轮廓）
        area = cv2.contourArea(cnt)
        if area < 10:  # 过滤极小面积轮廓（噪声）
            continue

        # 计算轮廓周长（用于圆形度计算）
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter < 10:  # 过滤极短周长轮廓
            continue

        # 计算圆形度（理想圆=1，值越高越接近圆）
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # 计算最小外接圆（获取半径）
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        # 筛选条件：圆形度达标 + 半径在范围内
        if (circularity > circularity_thresh and
                min_radius <= radius <= max_radius):
            balls.append({
                "contour": cnt,
                "center": (int(x), int(y)),
                "radius": int(radius),
                "circularity": round(circularity, 3)
            })

    # 6. 可视化检测结果
    img_result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for ball in balls:
        # 绘制球体轮廓（红色）
        cv2.drawContours(img_result, [ball["contour"]], -1, (0, 0, 255), 2)
        # 绘制球心（绿色点）
        cv2.circle(img_result, ball["center"], 3, (0, 255, 0), -1)
        # 标注半径和圆形度（蓝色文本）
        text = f"r={ball['radius']}, c={ball['circularity']}"
        cv2.putText(
            img_result, text,
            (ball["center"][0] + 10, ball["center"][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1
        )

    cvshow(f"球体检测结果（共{len(balls)}个球体）", img_result)
    return img_result, balls


if __name__ == "__main__":
    # 调用示例（替换为你的图像路径）
    try:
        result_img, detected_balls = detect_balls(
            image_path="data/bga.jpg",
            min_radius=9,  # 可根据实际球体大小调整
            max_radius=25,
            circularity_thresh=0.83  # 可根据球体圆润程度调整
        )
        print(f"检测完成，共发现{len(detected_balls)}个球体")
    except Exception as e:
        print(f"检测失败：{str(e)}")