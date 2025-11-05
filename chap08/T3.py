import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def visualize_matching():
    MIN_MATCH_COUNT = 10  # 最小匹配点数阈值
    # 读取图像（灰度模式）
    img1 = cv.imread('data/box.png', 0)          # 模板图像（待匹配的目标）
    img2 = cv.imread('data/box_in_scene.png', 0) # 场景图像（包含目标的图像）

    # 检查图像是否读取成功
    if img1 is None or img2 is None:
        raise FileNotFoundError("请检查图像路径是否正确！")

    # 1. 初始化SIFT特征检测器（提取关键点和描述符）
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)  # 模板图像的关键点和描述符
    kp2, des2 = sift.detectAndCompute(img2, None)  # 场景图像的关键点和描述符

    # 2. 可视化特征点
    def draw_keypoints(img, kp, title):
        """绘制关键点并显示"""
        kp_img = cv.drawKeypoints(
            img, kp, None,
            color=(0, 255, 0),  # 绿色标记关键点
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 显示关键点大小和方向
        )
        plt.figure(figsize=(8, 6))
        plt.imshow(cv.cvtColor(kp_img, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    draw_keypoints(img1, kp1, "模板图像的SIFT特征点")
    draw_keypoints(img2, kp2, "场景图像的SIFT特征点")

    # 3. 特征匹配（使用FLANN匹配器加速匹配）
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 索引参数
    search_params = dict(checks=50)  # 搜索参数（值越大越精确，速度越慢）
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # KNN匹配（k=2，返回每个特征点的前2个最佳匹配）
    matches = flann.knnMatch(des1, des2, k=2)

    # 4. 筛选优质匹配点（Lowe's比率测试）
    good_matches = []
    for m, n in matches:
        # 如果最佳匹配距离 < 次佳匹配距离的70%，则视为优质匹配
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"原始匹配点数：{len(matches)}，优质匹配点数：{len(good_matches)}")

    # 5. 若优质匹配点足够，计算单应性矩阵并定位目标
    if len(good_matches) > MIN_MATCH_COUNT:
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 模板图像中的点
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 场景图像中的对应点

        # 计算单应性矩阵（通过RANSAC算法剔除异常值）
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()  # 有效匹配点的掩码

        # 6. 定位模板在场景中的位置（绘制边界框）
        h, w = img1.shape  # 模板图像的高和宽
        # 模板图像的四个角点（左上角、左下角、右下角、右上角）
        template_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        # 通过单应性矩阵转换角点坐标到场景图像中
        scene_corners = cv.perspectiveTransform(template_corners, M)

        # 在场景图像上绘制定位框（绿色，线宽3）
        result_img = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)  # 转为彩色图以便标记
        result_img = cv.polylines(
            result_img,
            [np.int32(scene_corners)],  # 转换为整数坐标
            isClosed=True,  # 闭合多边形
            color=(0, 255, 0),
            thickness=3,
            lineType=cv.LINE_AA
        )

        # 显示定位结果
        plt.figure(figsize=(10, 8))
        plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
        plt.title("目标定位结果")
        plt.axis('off')
        plt.show()

    else:
        print(f"匹配点不足！需要至少{MIN_MATCH_COUNT}个，实际找到{len(good_matches)}个")
        matches_mask = None

    # 7. 可视化匹配的特征点对
    draw_params = dict(
        matchColor=(0, 255, 0),  # 匹配线为绿色
        singlePointColor=None,    # 不单独标记未匹配的点
        matchesMask=matches_mask, # 只显示有效匹配点
        flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS  # 不绘制未匹配的点
    )

    # 绘制匹配结果
    match_img = cv.drawMatches(
        img1, kp1,    # 模板图像及其关键点
        img2, kp2,    # 场景图像及其关键点
        good_matches, # 优质匹配点
        None,         # 输出图像（None则自动创建）
        **draw_params
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
    plt.title("特征匹配结果（绿色线为有效匹配）")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_matching()