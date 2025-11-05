# 导入必要的库
import argparse  # 用于解析命令行参数
import cv2.dnn  # OpenCV的深度学习模块，用于加载和运行ONNX模型
import numpy as np  # 用于数值计算和数组操作

from ultralytics.utils import ROOT  # 从ultralytics库导入ROOT路径工具

import yaml  # 用于解析YAML文件（存储类别名称）


# 定义YAML文件加载函数
def yaml_load(filepath):
    """
    加载并解析YAML文件内容

    参数:
        filepath: YAML文件的路径

    返回:
        解析后的YAML数据（通常为字典或列表）
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)  # 使用safe_load确保安全解析


# 从ultralytics库导入检查YAML文件的工具
from ultralytics.utils.checks import check_yaml  # 用于检查YAML文件是否存在并返回正确路径

# 加载COCO128数据集的类别名称
# check_yaml确保文件存在，yaml_load解析文件，['names']获取类别列表
CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']

# 为每个类别生成随机颜色（用于绘制边界框）
# 形状为(类别数, 3)，每个元素是0-255的随机值（BGR颜色空间）
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    在图像上绘制边界框和类别标签

    参数:
        img: 要绘制的图像
        class_id: 目标类别ID（对应CLASSES列表的索引）
        confidence: 检测置信度（0-1之间）
        x, y: 边界框左上角坐标
        x_plus_w, y_plus_h: 边界框右下角坐标
    """
    # 生成标签文本（类别名称 + 置信度）
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    # 获取该类别的颜色
    color = colors[class_id]

    # 绘制矩形边界框
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # 绘制标签文本（左上角偏移一点，避免覆盖边界框）
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    """
    主函数：加载模型、处理图像、执行检测并显示结果

    参数:
        onnx_model: ONNX格式的模型文件路径
        input_image: 输入图像的路径
    """
    # 加载ONNX模型
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # 读取输入图像
    original_image: np.ndarray = cv2.imread(input_image)
    # 获取图像的高、宽
    [height, width, _] = original_image.shape

    # 计算最长边（用于生成正方形画布，保持图像比例）
    length = max((height, width))
    # 创建一个正方形的空白画布（填充0，即黑色）
    image = np.zeros((length, length, 3), np.uint8)
    # 将原始图像放置在画布左上角（保持原始比例，避免拉伸）
    image[0:height, 0:width] = original_image
    # 计算缩放比例（将图像缩放到640x640，YOLOv8的输入尺寸）
    scale = length / 640

    # 对图像进行预处理：转换为blob格式
    # 1/255: 归一化到0-1范围
    # size=(640, 640): 调整大小
    # swapRB=True: 将RGB转换为BGR（OpenCV默认格式）
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    # 将预处理后的图像输入模型
    model.setInput(blob)
    # 执行推理，获取输出结果
    outputs = model.forward()

    # 转换输出格式（转置以方便解析）
    outputs = np.array([cv2.transpose(outputs[0])])
    # 获取输出的行数（检测到的候选框数量）
    rows = outputs.shape[1]

    # 存储检测到的边界框、置信度和类别ID
    boxes = []
    scores = []
    class_ids = []

    # 遍历所有候选框
    for i in range(rows):
        # 获取当前候选框的类别置信度（从第4个元素开始，前4个是边界框坐标）
        classes_scores = outputs[0][i][4:]
        # 找到最大置信度及其对应的类别索引
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)

        # 如果最大置信度大于阈值（0.25），保留该检测结果
        if maxScore >= 0.25:
            # 计算边界框坐标（中心点x, y, 宽, 高）
            # 转换为左上角坐标 (x - w/2, y - h/2)
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # 左上角x
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # 左上角y
                outputs[0][i][2],  # 宽度
                outputs[0][i][3]  # 高度
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # 执行非极大值抑制（NMS），去除重叠的冗余边界框
    # 参数：边界框列表、置信度列表、置信度阈值、NMS阈值、迭代次数
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    # 存储最终的检测结果
    detections = []
    # 遍历NMS筛选后的边界框
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]

        # 整理检测结果为字典
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale
        }
        detections.append(detection)

        # 在原始图像上绘制边界框
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),  # 左上角x（缩放回原始尺寸）
            round(box[1] * scale),  # 左上角y（缩放回原始尺寸）
            round((box[0] + box[2]) * scale),  # 右下角x（缩放回原始尺寸）
            round((box[1] + box[3]) * scale)  # 右下角y（缩放回原始尺寸）
        )

    # 显示检测结果图像
    cv2.imshow('image', original_image)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()  # 关闭所有窗口

    return detections


# 程序入口
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加模型路径参数（默认yolov8n.onnx）
    parser.add_argument('--model', default='yolov8n.onnx', help='Input your onnx model.')
    # 添加图像路径参数（默认data/bus.jpg）
    parser.add_argument('--img', default='data/bus.jpg', help='Path to input image.')
    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数
    main(args.model, args.img)