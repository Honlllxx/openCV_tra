import cv2
import numpy as np
import matplotlib.pyplot as plt


import matplotlib.font_manager as fm

# 1. 手动指定系统中已有的中文字体路径（Windows默认路径）
# 可选字体：微软雅黑（msyh.ttc）、宋体（simsun.ttc）、黑体（simhei.ttf）
font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑路径（Windows通用）
# font_path = "C:/Windows/Fonts/simsun.ttc"  # 宋体路径（备选）

# 2. 加载字体并设置全局字体
my_font = fm.FontProperties(fname=font_path, size=10)  # size为字体大小
plt.rcParams['font.family'] = my_font.get_name()  # 应用字体到全局
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 1. 读取图像（灰度模式）
img = cv2.imread('data/img.png', 0)  # 替换为你的图像路径
if img is None:
    raise ValueError("无法读取图像，请检查文件路径是否正确")
h, w = img.shape

# 2. 傅里叶变换与中心化
f = np.fft.fft2(img)  # 二维傅里叶变换
f_shift = np.fft.fftshift(f)  # 频域平移（低频移至中心）

# 3. 生成高斯低通和高通滤波器
def gaussian_filter(h, w, sigma, is_lowpass):
    # 生成频域坐标（以中心为原点）
    x = np.linspace(-w//2, w//2 - 1, w)
    y = np.linspace(-h//2, h//2 - 1, h)
    xx, yy = np.meshgrid(x, y)
    # 高斯函数
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian if is_lowpass else 1 - gaussian  # 低通/高通切换

# 4. 应用滤波器
sigma = 30  # 低通滤波标准差（越大越模糊）
glpf = gaussian_filter(h, w, sigma, is_lowpass=True)  # 低通滤波器
ghpf = gaussian_filter(h, w, sigma, is_lowpass=False)  # 高通滤波器

# 频域滤波（相乘）
f_lowpass = f_shift * glpf
f_highpass = f_shift * ghpf

# 5. 逆傅里叶变换（返回空间域）
f_lowpass_shift = np.fft.ifftshift(f_lowpass)  # 逆平移
img_lowpass = np.fft.ifft2(f_lowpass_shift).real  # 逆变换（取实部，消除虚部误差）
img_lowpass = np.clip(img_lowpass, 0, 255).astype(np.uint8)  # 归一化到0-255

f_highpass_shift = np.fft.ifftshift(f_highpass)
img_highpass = np.fft.ifft2(f_highpass_shift).real
img_highpass = np.clip(img_highpass, 0, 255).astype(np.uint8)

# 6. 结果显示 - 频域滤波
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('原始图像'), plt.axis('off')
plt.subplot(132), plt.imshow(img_lowpass, cmap='gray'), plt.title('高斯低通滤波（模糊）'), plt.axis('off')
plt.subplot(133), plt.imshow(img_highpass, cmap='gray'), plt.title('高斯高通滤波（边缘）'), plt.axis('off')
plt.tight_layout()
plt.show()

# 边缘检测部分
# 1. 读取图像（灰度模式）并去噪
img_blur = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯去噪（3x3核）

# 2. Sobel算子（x方向+y方向）
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, dx=1, dy=0, ksize=3)  # x方向（水平边缘）
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, dx=0, dy=1, ksize=3)  # y方向（垂直边缘）
sobel_abs = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))  # 梯度模长（绝对值）

# 3. Laplacian算子
laplacian = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)  # 二阶导数
laplacian_abs = cv2.convertScaleAbs(laplacian)  # 取绝对值，避免负梯度丢失

# 4. Canny算子
canny = cv2.Canny(img_blur, threshold1=50, threshold2=150)

# 5. 结果显示 - 边缘检测
plt.figure(figsize=(15, 5))
plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('原始图像'), plt.axis('off')
plt.subplot(142), plt.imshow(sobel_abs, cmap='gray'), plt.title('Sobel边缘'), plt.axis('off')
plt.subplot(143), plt.imshow(laplacian_abs, cmap='gray'), plt.title('Laplacian边缘'), plt.axis('off')
plt.subplot(144), plt.imshow(canny, cmap='gray'), plt.title('Canny边缘'), plt.axis('off')
plt.tight_layout()
plt.show()

# 清晰度评价部分
# 1. 生成不同清晰度的图像（高斯模糊模拟失焦）
def generate_blurred_images(img, sigma_list):
    blurred_imgs = []
    for sigma in sigma_list:
        if sigma == 0:
            blurred_imgs.append(img)  # sigma=0：原始清晰图像
        else:
            # 高斯核大小通常取6*sigma+1（确保覆盖主要区域）
            ksize = 2 * int(3 * sigma) + 1
            blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
            blurred_imgs.append(blurred)
    return blurred_imgs

# 2. 实现Tenegrad清晰度评价函数
def tenegrad_clarity(img, threshold=20):
    # Step 1: Sobel算子计算x、y方向梯度
    sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)
    # Step 2: 计算梯度模长
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    # Step 3: 阈值筛选（只保留大于threshold的梯度，排除噪声）
    gradient_filtered = gradient_mag[gradient_mag > threshold]
    # Step 4: 计算评分（梯度平方和）
    clarity_score = np.sum(gradient_filtered ** 2)
    return clarity_score

# 3. 实验流程
# 定义sigma列表（模拟不同失焦程度：0→清晰，10→极模糊）
sigma_list = [0, 1, 2, 3, 5, 7, 10]
# 生成不同清晰度图像
blurred_imgs = generate_blurred_images(img, sigma_list)
# 计算每个图像的Tenegrad评分
clarity_scores = [tenegrad_clarity(img) for img in blurred_imgs]

# 4. 结果可视化 - 清晰度评价
# 4.1 显示不同清晰度图像
plt.figure(figsize=(14, 6))
for i, (sigma, img_blur) in enumerate(zip(sigma_list, blurred_imgs)):
    plt.subplot(2, 4, i+1), plt.imshow(img_blur, cmap='gray')
    plt.title(f'sigma={sigma}\n评分={int(clarity_scores[i])}'), plt.axis('off')
plt.tight_layout()

# 4.2 显示清晰度评分曲线（sigma vs 评分）
plt.figure(figsize=(8, 4))
plt.plot(sigma_list, clarity_scores, 'o-', linewidth=2, color='red')
plt.xlabel('高斯模糊sigma（失焦程度）', fontsize=12)
plt.ylabel('Tenegrad清晰度评分', fontsize=12)
plt.title('失焦程度与清晰度评分关系', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
