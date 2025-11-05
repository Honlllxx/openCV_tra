import dxcam
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time


class DirectXOverlayDetector:
    """
    使用DirectX的高性能屏幕覆盖检测器
    """

    def __init__(self, region=(0, 0, 1920, 1080), model_path='yolov8n.pt', confidence_threshold=0.5):
        # 初始化YOLO模型
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # 初始化DXCam相机（高性能截图）
        self.camera = dxcam.create(region=region)
        self.camera.start(target_fps=60, video_mode=True)

        # 加载类别名称
        self.class_names = self.get_class_names()
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

        self.running = False
        self.detection_results = []

        print("DirectX覆盖检测器已初始化")
        print("按Ctrl+C退出程序")

    def get_class_names(self):
        """获取类别名称"""
        try:
            if hasattr(self.model, 'names') and self.model.names:
                return self.model.names
        except:
            pass
        return [str(i) for i in range(1000)]

    def start_detection(self):
        """开始检测线程"""
        self.running = True
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def detection_loop(self):
        """检测循环"""
        while self.running:
            try:
                # 获取最新帧
                frame = self.camera.get_latest_frame()

                if frame is not None:
                    # YOLO检测
                    results = self.model(frame, verbose=False)
                    self.detection_results = results

            except Exception as e:
                print(f"检测错误: {e}")
                time.sleep(0.1)

    def stop(self):
        """停止检测"""
        self.running = False
        if hasattr(self, 'camera'):
            self.camera.stop()
        print("检测已停止")

    def get_current_detections(self):
        """获取当前检测结果"""
        return self.detection_results