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
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    # 获取该类别的颜色
    color = colors[class_id]

    # 绘制矩形边界框
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # 绘制标签文本（左上角偏移一点，避免覆盖边界框）
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


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
    """

    def __init__(self, screen_frame=(3200, 2000), region=(1.0, 1.0), window_name='YOLO Screen Detection',
                 exit_key=0x1B, save_results=False, save_folder='results'):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--region', type=tuple, default=region,
                                 help='截图范围；分别为x，y，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')
        self.parser_args = self.parser.parse_args()

        self.cap = mss.mss()  # 实例化mss

        self.screen_width = screen_frame[0]  # 屏幕的宽
        self.screen_height = screen_frame[1]  # 屏幕的高
        self.mouse_x, self.mouse_y = self.screen_width // 2, self.screen_height // 2  # 屏幕中心点坐标

        # 截图区域
        self.GAME_WIDTH, self.GAME_HEIGHT = int(self.screen_width * self.parser_args.region[0]), int(
            self.screen_height * self.parser_args.region[1])  # 宽高
        self.GAME_LEFT, self.GAME_TOP = int(0 + self.screen_width // 2 * (1. - self.parser_args.region[0])), int(
            0 + self.screen_height // 2 * (1. - self.parser_args.region[1]))  # 原点

        self.RESZIE_WIN_WIDTH, self.RESIZE_WIN_HEIGHT = self.screen_width // 4, self.screen_height // 4  # 显示窗口大小
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

        # 创建保存结果的文件夹
        if self.save_results and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def grab_screen_mss(self, monitor):
        # cap.grab截取图片，np.array将图片转为数组，cvtColor将BRGA转为BRG,去掉了透明通道
        return cv2.cvtColor(np.array(self.cap.grab(monitor)), cv2.COLOR_BGRA2BGR)

    def update_img(self, img):
        self.img = img

    def get_img(self):
        return self.img

    # 计算中心坐标
    def calculate_position(self, xyxy):
        c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
        center_x = int((c2[0] - c1[0]) / 2 + c1[0])
        center_y = int((c2[1] - c1[1]) / 2 + c1[1])
        return center_x, center_y

    def compress_image_with_pil(self, img_array, quality=85):
        """
        使用PIL压缩图像

        参数:
            img_array: 输入的图像数组
            quality: 压缩质量 (0-100)
        """
        # 将OpenCV的BGR格式转换为PIL能识别的RGB格式
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # 创建一个BytesIO对象，用于保存压缩后的图像数据
        byte_io = io.BytesIO()

        # 压缩图像
        pil_image.save(byte_io, format='JPEG', quality=quality)

        # 将BytesIO缓冲区的数据读取为ndarray
        compressed_img_array = np.array(Image.open(byte_io))

        # 转换回OpenCV的BGR格式
        compressed_cv2_img = cv2.cvtColor(compressed_img_array, cv2.COLOR_RGB2BGR)

        return compressed_cv2_img

    def process_detections(self, results, img):
        """
        处理YOLO检测结果并在图像上绘制

        参数:
            results: YOLO检测结果
            img: 原始图像

        返回:
            绘制了检测结果的图像
        """
        # 创建图像的副本用于绘制
        result_img = img.copy()

        # 遍历所有检测结果
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # 绘制边界框和标签
                    x1, y1, x2, y2 = map(int, xyxy)
                    draw_bounding_box(result_img, cls, conf, x1, y1, x2, y2)

        return result_img

    def run(self):
        i = 0
        print("开始屏幕目标检测...")
        print("按ESC键退出程序")

        while True:
            # 捕获显示器图像
            img = self.grab_screen_mss(self.monitor)

            # 压缩图像（可选，如果需要提高性能可以取消注释）
            # img = self.compress_image_with_pil(img, quality=85)

            # YOLO处理图像
            results = model(img)

            # 处理检测结果并在图像上绘制
            result_img = self.process_detections(results, img)

            # 保存结果图像（如果启用）
            if self.save_results:
                filename = os.path.join(self.save_folder, f'image_{i:06d}.jpg')
                cv2.imwrite(filename, result_img)
                i += 1

            # 绘制窗口显示实时画面
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 根据窗口大小设置图片大小
            cv2.resizeWindow(self.window_name, self.RESZIE_WIN_WIDTH, self.RESIZE_WIN_HEIGHT)
            cv2.imshow(self.window_name, result_img)

            # 显示检测信息
            for r in results:
                if len(r.boxes) > 0:
                    print(f"检测到 {len(r.boxes)} 个目标")

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
    save_folder='detection_results'  # 保存结果的文件夹
)
sc.run()