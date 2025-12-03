from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# 初始化OCR（固定参数）
ocr = PaddleOCR(use_angle_cls=True, lang="ch", drop_score=0.3, show_log=False)


# 修复：中文绘制函数（文字显示在文本框外下方/右侧）
def putText_Chinese(img, strText, box, color, fontSize):
    """
    :param box: 完整文本框坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :param pos: 仅兼容旧参数，实际用box计算位置
    """
    try:
        # 强制使用Windows宋体（绝对路径）
        font = ImageFont.truetype("C:/Windows/Fonts/simfang.ttf", fontSize)
        # BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)

        # 提取文本框关键坐标
        x1, y1 = box[0][0], box[0][1]  # 框左上角
        x4, y4 = box[3][0], box[3][1]  # 框左下角
        box_width = abs(x4 - x1)  # 框宽度
        box_height = abs(y4 - y1)  # 框高度

        # ========== 核心：框外位置计算 ==========
        # 方案1：文本框正下方（推荐）- 框外10px，水平居中
        draw_x = int(x1 + box_width / 2 - (len(strText) * fontSize / 4))  # 文字水平居中
        draw_y = int(y4 + 10)  # 框下方10px

        # 方案2：文本框右侧（可选）- 框外10px，垂直居中
        # draw_x = int(x1 + box_width + 10)
        # draw_y = int(y1 + box_height / 2 + fontSize / 2)

        # 确保文字不超出图像边界
        draw_x = max(0, min(draw_x, img.shape[1] - len(strText) * fontSize))
        draw_y = max(fontSize, min(draw_y, img.shape[0] - 10))

        draw.text((draw_x, draw_y), strText, font=font, fill=color)
        # RGB转回BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"字体问题降级绘制：{e}")
        # 降级绘制也适配框外位置
        x1, y1 = box[0][0], box[0][1]
        x4, y4 = box[3][0], box[3][1]
        box_width = abs(x4 - x1)
        draw_x = int(x1 + box_width / 2 - (len(strText) * 8))  # 适配默认字体
        draw_y = int(y4 + 10)
        cv2.putText(img, strText, (draw_x, draw_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img


# 关键修复：暴力解析OCR结果（无视层级，直接提取所有文本框）
def force_draw_ocr(img_path):
    # 1. 读取图像（强制确认路径）
    img = cv2.imread(img_path)
    if img is None:
        print(f"图像读取失败：{img_path}")
        return None

    # 2. 执行OCR并打印原始结果（调试）
    result = ocr.ocr(img_path, cls=True)
    print(f"OCR原始结果（前2行）：{result[:2]}")

    # 3. 暴力遍历所有层级，提取文本框和文本
    all_boxes = []
    all_texts = []

    # 递归遍历所有嵌套列表
    def extract_boxes_text(obj):
        if isinstance(obj, list):
            for item in obj:
                # 匹配文本框格式：[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                if (isinstance(item, list) and len(item) == 4 and
                        isinstance(item[0], list) and len(item[0]) == 2):
                    all_boxes.append(item)
                # 匹配文本格式：(文本, 置信度) 或 [文本, 置信度]
                elif (isinstance(item, (list, tuple)) and len(item) == 2 and
                      isinstance(item[0], str) and isinstance(item[1], float)):
                    all_texts.append(item[0])
                extract_boxes_text(item)

    extract_boxes_text(result)

    # 4. 打印提取结果（确认命中）
    print(f"提取到 {len(all_boxes)} 个文本框，{len(all_texts)} 行文本")

    # 5. 强制绘制（一一对应）
    for i in range(min(len(all_boxes), len(all_texts))):
        box = all_boxes[i]
        text = all_texts[i]
        # 转换坐标为整数（用于画框）
        pt1 = (int(box[0][0]), int(box[0][1]))
        pt2 = (int(box[1][0]), int(box[1][1]))
        pt3 = (int(box[2][0]), int(box[2][1]))
        pt4 = (int(box[3][0]), int(box[3][1]))

        # 绘制边框（浅红，1px）
        cv2.line(img, pt1, pt2, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.line(img, pt2, pt3, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.line(img, pt3, pt4, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.line(img, pt4, pt1, (100, 200, 255), 1, cv2.LINE_AA)

        # 绘制文字（红色，21号字体，框外下方）
        img = putText_Chinese(img, text, box, (255, 0, 0), 21)

    return img


# ---------------------- 执行绘制 ----------------------
if __name__ == "__main__":
    # 替换为你的图像绝对路径（必须！）
    IMG_PATH = r".\imgs\temp2.png"

    # 强制绘制
    drawn_img = force_draw_ocr(IMG_PATH)

    if drawn_img is not None:
        # 缩放图像到屏幕可见大小
        drawn_img = cv2.resize(drawn_img, (1000, 800), cv2.INTER_AREA)

        # 显示+保存
        cv2.imshow("FORCE-OCR-Result", drawn_img)
        cv2.imwrite("force_drawn_result.png", drawn_img)
        print("\n强制绘制完成！请查看 force_drawn_result.png")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("绘制失败")