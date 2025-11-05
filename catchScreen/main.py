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

#实例化YOLO类，加载模型（D:\Python\Lib\site-packages\ultralytics\models\yolo\model.py）
model = YOLO('D:\\Python project\\openCV_tra\\30980-02\chp9\\yolov8n.pt')
#捕捉屏幕
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
    """

    def __init__(self, screen_frame=(3200, 2000), region=(1.0, 1.0), window_name='YOLO', exit_key=0x1B):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--region', type=tuple, default=region,
                                 help='截图范围；分别为x，y，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')
        self.parser_args = self.parser.parse_args()

        #self.cap = mss.mss(mon=-1, optimize=True)
        self.cap = mss.mss()# 实例化mss，并使用高效模式

        self.screen_width = screen_frame[0]  # 屏幕的宽
        self.screen_height = screen_frame[1]  # 屏幕的高
        self.mouse_x, self.mouse_y = self.screen_width // 2, self.screen_height // 2  # 屏幕中心点坐标

        # 截图区域
        self.GAME_WIDTH, self.GAME_HEIGHT = int(self.screen_width * self.parser_args.region[0]), int(
            self.screen_height * self.parser_args.region[1])  # 宽高
        self.GAME_LEFT, self.GAME_TOP = int(0 + self.screen_width // 2 * (1. - self.parser_args.region[0])), int(
            0 + 1080 // 2 * (1. - self.parser_args.region[1]))  # 原点

        self.RESZIE_WIN_WIDTH, self.RESIZE_WIN_HEIGHT = self.screen_width // 4, self.screen_height // 4  # 显示窗口大小
        self.mointor = {
            'left': self.GAME_LEFT,
            'top': self.GAME_TOP,
            'width': self.GAME_WIDTH,
            'height': self.GAME_HEIGHT
        }

        self.window_name = window_name
        self.Exit_key = exit_key
        self.img = None

    def grab_screen_mss(self, monitor):
        # cap.grab截取图片，np.array将图片转为数组，cvtColor将BRGA转为BRG,去掉了透明通道
        return cv2.cvtColor(np.array(self.cap.grab(monitor)), cv2.COLOR_BGRA2BGR)

    def update_img(self, img):
        self.img = img

    def get_img(self):
        return self.img

    #计算中心坐标
    def calculate_position(xyxy):
        c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
        center_x = int((c2[0] - c1[0]) / 2 + c1[0])
        center_y = int((c2[1] - c1[1]) / 2 + c1[1])
        return center_x, center_y
    
    def compress_image_with_pil(self, img_array):
        # 将OpenCV的BGR格式转换为PIL能识别的RGB格式
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # 创建一个BytesIO对象，用于保存压缩后的图像数据
        byte_io = io.BytesIO()

        # 假设我们将其压缩到原质量的50%，您可以根据需求调整质量参数
        pil_image.save(byte_io, format='JPEG', quality=50)

        # 将BytesIO缓冲区的数据读取为ndarray
        compressed_img_array = np.array(Image.open(byte_io))

        # 转换回OpenCV的BGR格式
        compressed_cv2_img = cv2.cvtColor(compressed_img_array, cv2.COLOR_RGB2BGR)

        return compressed_cv2_img

    def run(self):
        i = 0
        while True:
            #捕获显示器图像
            if self.img is None:
                img = self.grab_screen_mss(self.mointor)
            
            # 压缩图像
            img = self.compress_image_with_pil(img)

            #YOLO处理图像
            results = model(img)

            # 在原始截图上绘制检测结果
            for r in results:
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])  
                #im.show()  # show image
                savefolder = 'results'
                
                filename = os.path.join(savefolder,'image_{}.jpg'.format(i))
                im.save(filename)
                i = i+1
                
            #绘制窗口显示实时画面
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL 根据窗口大小设置图片大小
            cv2.resizeWindow(self.window_name, self.RESZIE_WIN_WIDTH, self.RESIZE_WIN_HEIGHT)
            cv2.imshow(self.window_name, img)

            # 结束键 ESC
            if cv2.waitKey(1) & 0xff == self.Exit_key:  
                cv2.destroyAllWindows()
                print("运行结束")
                exit("结束")

#检查yolo库（Ultralytics YOLOv8.1.40 🚀把 Python-3.12.1 torch-2.2.1+cpu CPU (12th Gen Intel Core(TM) i5-12400)
#Setup complete ✅ (12 CPUs, 15.7 GB RAM, 100.7/195.3 GB disk)）
ultralytics.checks()
print("开始运行")
#pyautogui.click(center_x*1920,center_y*1080, duration=0.1)
sc = ScreenCapture()
sc.run()
    
    
