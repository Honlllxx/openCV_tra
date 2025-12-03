import dxcam
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import signal
import sys


class DirectXOverlayDetector:
    """
    使用DirectX的高性能屏幕覆盖检测器 - 在屏幕上实时绘制检测框
    """

    def __init__(self, region=(0, 0, 1920, 1080), model_path='yolov8n.pt', confidence_threshold=0.5):
        # 初始化YOLO模型
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # 屏幕区域设置
        self.region = region
        self.screen_width = region[2] - region[0]
        self.screen_height = region[3] - region[1]

        # 初始化DXCam相机（高性能截图）
        self.camera = dxcam.create(region=region)
        self.camera.start(target_fps=60, video_mode=True)

        # 加载类别名称
        self.class_names = self.get_class_names()
        self.colors  = np.random.uniform(0, 255, size=(len(self.class_names), 3))

        # 检测状态
        self.running = False
        self.current_detections = []
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        # 创建覆盖窗口
        self.create_overlay_window()

        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)

        print("DirectX覆盖检测器已初始化")
        print(f"检测区域: {region}")
        print("按Ctrl+C退出程序")

    def get_class_names(self):
        """获取类别名称"""
        try:
            if hasattr(self.model, 'names') and self.model.names:
                return self.model.names
        except:
            pass
        return [str(i) for i in range(1000)]

    def create_overlay_window(self):
        """创建透明覆盖窗口"""
        self.root = tk.Tk()
        self.root.title("实时目标检测覆盖")

        # 设置窗口属性
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)  # 始终在最前面
        self.root.attributes('-transparentcolor', 'black')  # 设置黑色为透明色
        self.root.configure(bg='black')  # 背景设为黑色
        self.root.overrideredirect(True)  # 无边框窗口

        # 创建画布
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            highlightthickness=0,
            bg='black'
        )
        self.canvas.pack()

        # 绑定退出事件
        self.root.bind('<Escape>', lambda e: self.stop())
        self.root.bind('<Control-c>', lambda e: self.stop())

    def capture_screen(self):
        """捕获屏幕截图"""
        try:
            frame = self.camera.get_latest_frame()
            if frame is not None:
                # DXCam返回BGR格式，转换为RGB
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"截图错误: {e}")
        return None

    def process_detections(self, results):
        """处理检测结果"""
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = box.conf.item()

                    if confidence < self.confidence_threshold:
                        continue

                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls.item())
                    class_name = self.class_names[class_id]

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'color': self.colors[class_id]
                    })

        return detections

    def draw_detections_on_canvas(self, detections):
        """在画布上绘制检测结果"""
        # 清空之前的绘制
        self.canvas.delete("all")

        # 绘制检测框和标签
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            hex_color = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))

            # 绘制边界框
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=hex_color,
                width=3,
                tags="detection"
            )

            # 绘制标签背景
            label = f"{det['class_name']} {det['confidence']:.2f}"
            text_id = self.canvas.create_text(
                x1, y1 - 5,
                text=label,
                fill=hex_color,
                anchor='sw',
                font=('Arial', 12, 'bold'),
                tags="label"
            )

            # 添加白色边框使文字更清晰
            bbox = self.canvas.bbox(text_id)
            self.canvas.create_rectangle(
                bbox,
                outline='white',
                fill='black',
                width=1,
                tags="label_bg"
            )
            self.canvas.tag_raise(text_id, "label_bg")

        # 显示FPS和检测数量
        info_text = f"FPS: {self.fps:.1f} | 检测目标: {len(detections)} 个"
        self.canvas.create_text(
            10, 10,
            text=info_text,
            fill='yellow',
            anchor='nw',
            font=('Arial', 14, 'bold'),
            tags="info"
        )

    def update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time

    def detection_loop(self):
        """检测循环"""
        while self.running:
            try:
                # 捕获屏幕
                frame = self.capture_screen()

                if frame is not None:
                    # YOLO检测
                    results = self.model(frame, verbose=False)

                    # 处理检测结果
                    self.current_detections = self.process_detections(results)

                    # 更新FPS
                    self.update_fps()

                    # 在画布上绘制检测结果
                    self.draw_detections_on_canvas(self.current_detections)

                    # 打印检测信息（每秒一次）
                    if self.frame_count == 0:
                        print(f"FPS: {self.fps:.1f} | 检测目标: {len(self.current_detections)} 个")
                        for det in self.current_detections:
                            print(f"  - {det['class_name']}: {det['confidence']:.2f}")

                # 短暂休眠以避免过高CPU占用
                time.sleep(0.01)

            except Exception as e:
                print(f"检测错误: {e}")
                time.sleep(0.1)

    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n正在退出...")
        self.stop()

    def start_detection(self):
        """开始检测"""
        self.running = True

        # 启动检测线程
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        print("开始实时目标检测覆盖...")

        # 启动GUI主循环
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"GUI错误: {e}")
        finally:
            self.stop()

    def stop(self):
        """停止检测"""
        self.running = False
        if hasattr(self, 'camera'):
            self.camera.stop()
        if hasattr(self, 'root'):
            self.root.quit()
            self.root.destroy()
        print("检测已停止")


# 简化版本 - 只在控制台输出，不显示覆盖
class ConsoleOnlyDetector:
    """只在控制台输出检测结果的版本"""

    def __init__(self, region=(0, 0, 1920, 1080), model_path='yolov8n.pt', confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # 初始化DXCam
        self.camera = dxcam.create(region=region)
        self.camera.start(target_fps=30, video_mode=True)

        self.class_names = self.get_class_names()
        self.running = False

        print("控制台检测器已初始化")
        print("按Ctrl+C退出程序")

    def get_class_names(self):
        try:
            if hasattr(self.model, 'names') and self.model.names:
                return self.model.names
        except:
            pass
        return [str(i) for i in range(1000)]

    def run(self):
        """运行检测"""
        self.running = True
        frame_count = 0
        start_time = time.time()

        print("开始后台目标检测...")

        try:
            while self.running:
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    results = self.model(frame, verbose=False)

                    # 处理并显示结果
                    detections = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                if box.conf.item() >= self.confidence_threshold:
                                    class_id = int(box.cls.item())
                                    detections.append({
                                        'class': self.class_names[class_id],
                                        'confidence': box.conf.item()
                                    })

                    # 每秒输出一次结果
                    frame_count += 1
                    if time.time() - start_time >= 1.0:
                        fps = frame_count / (time.time() - start_time)
                        print(f"FPS: {fps:.1f} | 检测到 {len(detections)} 个目标")
                        for det in detections:
                            print(f"  - {det['class']}: {det['confidence']:.2f}")
                        print("-" * 40)

                        frame_count = 0
                        start_time = time.time()

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n检测被用户中断")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            self.camera.stop()
            print("检测结束")


# 使用示例
def main():
    # 方法1: 使用屏幕覆盖（在屏幕上直接绘制检测框）
    detector = DirectXOverlayDetector(
        region=(0, 0, 1920, 1080),  # 检测区域
        model_path='../30980-02/chp9/yolov8n.pt',
        confidence_threshold=0.5
    )
    detector.start_detection()

    # 方法2: 只在控制台输出（不显示覆盖）
    # detector = ConsoleOnlyDetector(
    #     region=(0, 0, 1920, 1080),
    #     model_path='yolov8n.pt',
    #     confidence_threshold=0.5
    # )
    # detector.run()


if __name__ == "__main__":
    main()