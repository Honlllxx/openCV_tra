from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 初始化PaddleOCR（仅需运行一次下载模型）
# 支持ch/en/fr/german/korean/japan等语言
ocr = PaddleOCR(use_angle_cls=True, lang="ch")


# OpenCV显示中文的辅助函数
def putText_Chinese(img, strText, pos, color, fontSize):
    try:
        # 字体路径请根据实际情况调整（确保simfang.ttf存在）
        fontpath = "data/fonts/simfang.ttf"
        font = ImageFont.truetype(fontpath, fontSize)
    except:
        # 备用字体路径（适配不同环境）
        fontpath = "C:/Windows/Fonts/simfang.ttf"
        font = ImageFont.truetype(fontpath, fontSize)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # 防止文字超出图像边界
    pos_x = max(0, pos[0])
    pos_y = max(fontSize, pos[1])
    draw.text((pos_x, pos_y - fontSize), strText, font=font, fill=color)
    img = np.array(img_pil)
    return img


# 第一部分：测试OCR识别（可选）
print('---------------基础OCR识别---------------------')
img_path1 = 'data/imgs/00056221.jpg'
result1 = ocr.ocr(img_path1, cls=True)
for idx in range(len(result1)):
    res = result1[idx]
    for line in res:
        if line is not None and len(line) >= 2:  # 增加非空判断
            print(line)

# 第二部分：OpenCV可视化识别结果
print('---------------OpenCV 可视化---------------------')
img_path2 = 'data/sgl_ticket.jpg'
img = cv2.imread(img_path2)

# 检查图像是否成功加载
if img is None:
    print(f"错误：无法加载图像 {img_path2}，请检查路径是否正确")
else:
    cv2.imshow("src", img)
    result2 = ocr.ocr(img_path2, cls=True)

    # 遍历识别结果并绘制
    for idx in range(len(result2)):
        res = result2[idx]
        for line in res:
            # 核心修复：严格校验数据格式
            if (line is not None and isinstance(line, list) and
                    len(line) >= 2 and isinstance(line[0], list) and len(line[0]) >= 4):
                print(line)
                print('----------------------------')

                # 提取四个角点坐标并转换为整数
                box = line[0]  # 坐标列表 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                pt1 = (int(box[0][0]), int(box[0][1]))
                pt2 = (int(box[1][0]), int(box[1][1]))
                pt3 = (int(box[2][0]), int(box[2][1]))
                pt4 = (int(box[3][0]), int(box[3][1]))

                # 绘制文本框（红色边框）
                cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(img, pt2, pt3, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(img, pt3, pt4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(img, pt4, pt1, (0, 0, 255), 1, cv2.LINE_AA)

                # 绘制识别文本（紫红色）
                text = line[1][0] if len(line[1]) >= 1 else ""  # 提取识别文本
                img = putText_Chinese(img, text, pt1, (255, 0, 255), 20)

    # 显示和保存结果
    cv2.imshow("OCR-Result", img)
    cv2.imwrite("result.png", img)
    cv2.waitKey(0)  # 等待按键（0表示无限等待）
    cv2.destroyAllWindows()
from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 初始化PaddleOCR（仅需运行一次下载模型）
# 支持ch/en/fr/german/korean/japan等语言
ocr = PaddleOCR(use_angle_cls=True, lang="ch")


# OpenCV显示中文的辅助函数
def putText_Chinese(img, strText, pos, color, fontSize):
    try:
        # 字体路径请根据实际情况调整（确保simfang.ttf存在）
        fontpath = "data/fonts/simfang.ttf"
        font = ImageFont.truetype(fontpath, fontSize)
    except:
        # 备用字体路径（适配不同环境）
        fontpath = "C:/Windows/Fonts/simfang.ttf"
        font = ImageFont.truetype(fontpath, fontSize)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # 防止文字超出图像边界
    pos_x = max(0, pos[0])
    pos_y = max(fontSize, pos[1])
    draw.text((pos_x, pos_y - fontSize), strText, font=font, fill=color)
    img = np.array(img_pil)
    return img


# 第一部分：测试OCR识别（可选）
print('---------------基础OCR识别---------------------')
img_path1 = 'data/imgs/00056221.jpg'
result1 = ocr.ocr(img_path1, cls=True)
for idx in range(len(result1)):
    res = result1[idx]
    for line in res:
        if line is not None and len(line) >= 2:  # 增加非空判断
            print(line)

# 第二部分：OpenCV可视化识别结果
print('---------------OpenCV 可视化---------------------')
img_path2 = 'data/sgl_ticket.jpg'
img = cv2.imread(img_path2)

# 检查图像是否成功加载
if img is None:
    print(f"错误：无法加载图像 {img_path2}，请检查路径是否正确")
else:
    cv2.imshow("src", img)
    result2 = ocr.ocr(img_path2, cls=True)

    # 遍历识别结果并绘制
    for idx in range(len(result2)):
        res = result2[idx]
        for line in res:
            # 核心修复：严格校验数据格式
            if (line is not None and isinstance(line, list) and
                    len(line) >= 2 and isinstance(line[0], list) and len(line[0]) >= 4):
                print(line)
                print('----------------------------')

                # 提取四个角点坐标并转换为整数
                box = line[0]  # 坐标列表 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                pt1 = (int(box[0][0]), int(box[0][1]))
                pt2 = (int(box[1][0]), int(box[1][1]))
                pt3 = (int(box[2][0]), int(box[2][1]))
                pt4 = (int(box[3][0]), int(box[3][1]))

                # 绘制文本框（红色边框）
                cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(img, pt2, pt3, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(img, pt3, pt4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(img, pt4, pt1, (0, 0, 255), 1, cv2.LINE_AA)

                # 绘制识别文本（紫红色）
                text = line[1][0] if len(line[1]) >= 1 else ""  # 提取识别文本
                img = putText_Chinese(img, text, pt1, (255, 0, 255), 20)

    # 显示和保存结果
    cv2.imshow("OCR-Result", img)
    cv2.imwrite("result.png", img)
    cv2.waitKey(0)  # 等待按键（0表示无限等待）
    cv2.destroyAllWindows()