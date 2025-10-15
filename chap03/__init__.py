import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def add_salt_noise(image, prob):
    """添加盐噪声"""
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 255
            elif rdn > thres:
                output[i][j] = image[i][j]
            else:
                output[i][j] = image[i][j]
    return output


def add_pepper_noise(image, prob):
    """添加椒噪声"""
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = image[i][j]
            else:
                output[i][j] = image[i][j]
    return output


def add_salt_pepper_noise(image, prob):
    """添加椒盐噪声"""
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def add_gaussian_noise(image, mean=0, var=0.001):
    """添加高斯噪声"""
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out * 255)
    return out


def add_speckle_noise(image, var=0.04):
    """添加斑点噪声"""
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, var ** 0.5, image.shape)
    out = image + image * noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out * 255)
    return out


def apply_filters(image):
    """应用各种滤波"""
    # 均值滤波
    mean_blur = cv2.blur(image, (5, 5))

    # 方框滤波
    box_blur = cv2.boxFilter(image, -1, (5, 5), normalize=True)

    # 高斯滤波
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # 中值滤波
    median_blur = cv2.medianBlur(image, 5)

    # 双边滤波
    bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75)

    return mean_blur, box_blur, gaussian_blur, median_blur, bilateral_blur


def main():
    # 读取图像
    img = cv2.imread('data/img.png')
    if img is None:
        print("无法读取图像，请检查文件路径")
        return

    # 转换为RGB格式以便matplotlib显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 添加各种噪声
    salt_noise = add_salt_noise(img_rgb, 0.02)
    pepper_noise = add_pepper_noise(img_rgb, 0.02)
    salt_pepper_noise = add_salt_pepper_noise(img_rgb, 0.02)
    gaussian_noise = add_gaussian_noise(img_rgb, 0, 0.005)
    speckle_noise = add_speckle_noise(img_rgb, 0.04)

    # 显示原图和各种噪声图像
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('原图')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(salt_noise)
    plt.title('盐噪声')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(pepper_noise)
    plt.title('椒噪声')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(salt_pepper_noise)
    plt.title('椒盐噪声')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(gaussian_noise)
    plt.title('高斯噪声')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(speckle_noise)
    plt.title('斑点噪声')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 对带高斯噪声的图像应用高斯滤波
    gaussian_filtered = cv2.GaussianBlur(gaussian_noise, (5, 5), 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gaussian_noise)
    plt.title('带高斯噪声的图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gaussian_filtered)
    plt.title('高斯滤波后的图像')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 对带椒盐噪声的图像应用中值滤波
    median_filtered = cv2.medianBlur(salt_pepper_noise, 5)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(salt_pepper_noise)
    plt.title('带椒盐噪声的图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(median_filtered)
    plt.title('中值滤波后的图像')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 对原图应用各种滤波并显示
    mean_blur, box_blur, gaussian_blur, median_blur, bilateral_blur = apply_filters(img_rgb)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('原图')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(mean_blur)
    plt.title('均值滤波')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(box_blur)
    plt.title('方框滤波')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(gaussian_blur)
    plt.title('高斯滤波')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(median_blur)
    plt.title('中值滤波')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(bilateral_blur)
    plt.title('双边滤波')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
