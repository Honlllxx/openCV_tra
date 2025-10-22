import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


# 1. 轮廓提取与可视化
def contour_detection(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径")

    # 转为灰度图
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 阈值化处理（使用代码中指定的127阈值）
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    # 提取轮廓（保持RETR_TREE模式）
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上绘制轮廓
    img_with_contours = img.copy()
    # 用绿色绘制所有轮廓，线宽2
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    # 显示原图与轮廓结果
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("原图")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("轮廓提取结果")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return img


# 2. 分水岭分割实现（严格遵循提供的代码流程）
def watershed_segmentation(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径")

    # 显示原图（OpenCV窗口）
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input image", img)

    # 均值迁移滤波预处理
    blurred = cv2.pyrMeanShiftFiltering(img, sp=10, sr=100)

    # 转为灰度图并二值化（OTSU阈值）
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary-image", binary)

    # 形态学操作（开运算+膨胀）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(mb, kernel, iterations=3)
    cv2.imshow("确定背景区域", sure_bg)

    # 距离变换
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
    dist_output = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow("距离变换图", dist_output * 50)  # 放大亮度便于显示

    # 确定前景区域
    ret, surface = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(surface)
    cv2.imshow("确定前景区域", surface_fg)

    # 计算未知区域
    unknown = cv2.subtract(sure_bg, surface_fg)

    # 标记连通组件
    ret, markers = cv2.connectedComponents(surface_fg)
    markers = markers + 1  # 背景标记偏移
    markers[unknown == 255] = 0  # 未知区域标记为0

    # 打印标记信息（调试用）
    print("标记数据类型:", type(markers))
    print("分水岭前标记值为2的像素数:", np.sum(markers == 2))

    # 应用分水岭算法
    markers = cv2.watershed(img, markers=markers)
    print("分水岭后所有标记值:", np.unique(markers))
    print("分水岭后标记值为2的像素数:", np.sum(markers == 2))

    # 标记分割边界
    img[markers == -1] = [0, 0, 255]
    cv2.imshow("分水岭分割结果", img)

    # 生成并显示分割类别标记图（Matplotlib）
    markers_visual = markers + 1  # 调整标记值范围便于显示
    plt.figure(figsize=(8, 6))
    plt.imshow(markers_visual, cmap="tab20")  # 使用多色映射增强区分度
    plt.title("分割类别标记图")
    plt.axis("off")
    plt.show()

    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "./data/coins.jpg"  # 替换为你的图像路径
    # 1. 轮廓提取与显示
    contour_detection(image_path)
    # 2. 分水岭分割与显示
    watershed_segmentation(image_path)