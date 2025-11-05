import argparse
import time
import cv2
import keyboard
import mss
import numpy as np
import win32com.client
import win32con
import win32gui
from PIL import Image
import io
import os
from ultralytics import YOLO
import pyautogui
import ultralytics
import torch
import yaml
from ultralytics.utils import ROOT
from ultralytics.utils.checks import check_yaml

# 实例化YOLO类，加载模型
model = YOLO('D:\\Python project\\openCV_tra\\30980-02\\chp9\\yolov8n.pt')


# 加载COCO数据集的类别名称
def yaml_load(filepath):
    """加载并解析YAML文件内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# 尝试加载类别名称，如果失败则使用默认类别
try:
    CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
except:
    # 如果无法加载YAML文件，使用COCO数据集的基本类别
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

# 为每个类别生成随机颜色（用于绘制边界框）
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
    label = f'{CLASSES[class_id]} {confidence:.2f}'
    # 获取该类别的颜色
    color = colors[class_id]

    # 计算边界框的宽度和高度
    box_width = x_plus_w - x
    box_height = y_plus_h - y

    # 绘制半透明填充矩形作为背景
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x_plus_w, y_plus_h), color, -1)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

    # 绘制矩形边界框
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 3)

    # 计算文本大小
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # 绘制标签背景
    cv2.rectangle(img, (x, y - text_height - 10), (x + text_width, y), color, -1)
    # 绘制标签文本
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 在边界框中心绘制小圆点
    center_x = (x + x_plus_w) // 2
    center_y = (y + y_plus_h) // 2
    cv2.circle(img, (center_x, center_y), 4, color, -1)


def draw_detection_info(img, detections_count, fps):
    """
    在图像上绘制检测统计信息
    """
    info_text = f'Detections: {detections_count} | FPS: {fps:.1f}'
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


class ScreenCapture:
    """
    parameters
    ----------
        screen_frame : Tuple[int, int]
            屏幕宽高，分别为x，y
        region : Tuple[float, float]
            实际截图范围，分别为x，y，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)
        window_name : str
            显示窗口名
        exit_key : int
            结束窗口的退出键值，为键盘各键对应的ASCII码值，默认是ESC键
        save_results : bool
            是否保存检测结果图像
        save_folder : str
            保存结果的文件夹路径
        confidence_threshold : float
            检测置信度阈值
    """

    def __init__(self, screen_frame=(1920, 1080), region=(1.0, 1.0), window_name='YOLO Screen Detection',
                 exit_key=0x1B, save_results=False, save_folder='results', confidence_threshold=0.5):

        self.cap = mss.mss()  # 实例化mss

        self.screen_width = screen_frame[0]  # 屏幕的宽
        self.screen_height = screen_frame[1]  # 屏幕的高

        # 截图区域
        self.GAME_WIDTH, self.GAME_HEIGHT = int(self.screen_width * region[0]), int(
            self.screen_height * region[1])  # 宽高
        self.GAME_LEFT, self.GAME_TOP = int(0 + self.screen_width // 2 * (1. - region[0])), int(
            0 + self.screen_height // 2 * (1. - region[1]))  # 原点

        self.RESZIE_WIN_WIDTH, self.RESIZE_WIN_HEIGHT = 1200, 800  # 显示窗口大小
        self.monitor = {
            'left': self.GAME_LEFT,
            'top': self.GAME_TOP,
            'width': self.GAME_WIDTH,
            'height': self.GAME_HEIGHT
        }

        self.window_name = window_name
        self.Exit_key = exit_key
        self.img = None
        self.save_results = save_results
        self.save_folder = save_folder
        self.confidence_threshold = confidence_threshold

        # 创建保存结果的文件夹
        if self.save_results and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def grab_screen_mss(self, monitor):
        # cap.grab截取图片，np.array将图片转为数组，cvtColor将BRGA转为BRG,去掉了透明通道
        return cv2.cvtColor(np.array(self.cap.grab(monitor)), cv2.COLOR_BGRA2BGR)

    def process_detections(self, results, img):
        """
        处理YOLO检测结果并在图像上绘制

        参数:
            results: YOLO检测结果
            img: 原始图像

        返回:
            绘制了检测结果的图像和检测数量
        """
        # 创建图像的副本用于绘制
        result_img = img.copy()
        detections_count = 0

        # 遍历所有检测结果
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标和置信度
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # 只显示置信度高于阈值的目标
                    if conf >= self.confidence_threshold:
                        # 绘制边界框和标签
                        x1, y1, x2, y2 = map(int, xyxy)
                        draw_bounding_box(result_img, cls, conf, x1, y1, x2, y2)
                        detections_count += 1

        return result_img, detections_count

    def run(self):
        i = 0
        print("开始屏幕目标检测...")
        print("按ESC键退出程序")

        # FPS计算
        fps_counter = 0
        fps_time = time.time()
        fps = 0

        while True:
            start_time = time.time()

            # 捕获显示器图像
            img = self.grab_screen_mss(self.monitor)

            # YOLO处理图像
            results = model(img, verbose=False)  # verbose=False关闭控制台输出

            # 处理检测结果并在图像上绘制
            result_img, detections_count = self.process_detections(results, img)

            # 计算FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            # 绘制检测信息
            draw_detection_info(result_img, detections_count, fps)

            # 绘制检测区域边界
            cv2.rectangle(result_img, (0, 0), (self.GAME_WIDTH, self.GAME_HEIGHT), (0, 255, 0), 2)
            cv2.putText(result_img, 'Detection Area', (10, self.GAME_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 保存结果图像（如果启用）
            if self.save_results and detections_count > 0:
                filename = os.path.join(self.save_folder, f'image_{i:06d}.jpg')
                cv2.imwrite(filename, result_img)
                i += 1

            # 绘制窗口显示实时画面
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.RESZIE_WIN_WIDTH, self.RESIZE_WIN_HEIGHT)
            cv2.imshow(self.window_name, result_img)

            # 控制台输出检测信息（每秒输出一次）
            if fps_counter == 0 and detections_count > 0:
                print(f"检测到 {detections_count} 个目标 | FPS: {fps}")

            # 结束键 ESC
            if cv2.waitKey(1) & 0xff == self.Exit_key:
                cv2.destroyAllWindows()
                print("运行结束")
                break


# 检查yolo库
ultralytics.checks()
print("开始运行")

# 创建并运行屏幕捕获和检测
sc = ScreenCapture(
    screen_frame=(1920, 1080),  # 根据你的屏幕分辨率调整
    region=(0.8, 0.8),  # 检测区域大小
    save_results=True,  # 是否保存结果
    save_folder='detection_results',  # 保存结果的文件夹
    confidence_threshold=0.5  # 置信度阈值
)
sc.run()