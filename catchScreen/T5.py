# 预处理方式不同
#
# 后处理实现差异
#
# 置信度和NMS阈值设置
#
# 图像缩放和填充策略

import dxcam
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import tkinter as tk
import signal
import sys


class ImprovedDirectXOverlayDetector:
    """
    改进的高性能屏幕覆盖检测器 - 更准确的检测结果
    """

    def __init__(self, region=(0, 0, 1920, 1080), model_path='yolov8n.pt',
                 confidence_threshold=0.25, nms_threshold=0.45):
        # 初始化YOLO模型
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # 屏幕区域设置
        self.region = region
        self.screen_width = region[2] - region[0]
        self.screen_height = region[3] - region[1]

        # 初始化DXCam相机
        self.camera = dxcam.create(region=region)
        self.camera.start(target_fps=60, video_mode=True)

        # 加载类别名称
        self.class_names = self.get_class_names()
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

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

        print("改进版覆盖检测器已初始化")
        print(f"检测区域: {region}")
        print(f"置信度阈值: {confidence_threshold}, NMS阈值: {nms_threshold}")
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
        self.root.title("改进版实时目标检测")

        # 设置窗口属性
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'black')
        self.root.configure(bg='black')
        self.root.overrideredirect(True)

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
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"截图错误: {e}")
        return None

    def custom_preprocess(self, image):
        """
        自定义预处理 - 模仿ONNX版本的预处理流程
        """
        # 获取原始图像尺寸
        height, width = image.shape[:2]

        # 计算最长边（用于生成正方形画布）
        length = max(height, width)

        # 创建正方形画布
        padded_image = np.zeros((length, length, 3), dtype=np.uint8)
        padded_image[0:height, 0:width] = image

        # 计算缩放比例
        scale = length / 640.0

        # 调整大小到640x640
        resized = cv2.resize(padded_image, (640, 640))

        # 归一化
        normalized = resized.astype(np.float32) / 255.0

        # 转换通道顺序 BGR to RGB (如果需要)
        # normalized = normalized[..., ::-1]  # 如果模型需要RGB输入

        # 转换为模型需要的格式 (1, 3, 640, 640)
        blob = np.transpose(normalized, (2, 0, 1))
        blob = np.expand_dims(blob, 0)

        return blob, scale, (height, width)

    def custom_postprocess(self, outputs, scale, original_shape):
        """
        自定义后处理 - 模仿ONNX版本的后处理流程
        """
        detections = []

        # 转换输出格式
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # 解析输出
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)

            if maxScore >= self.confidence_threshold:
                # 计算边界框坐标 (中心点格式转角点格式)
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x1
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y1
                    outputs[0][i][2],  # width
                    outputs[0][i][3]  # height
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # 应用NMS
        if len(boxes) > 0:
            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold,
                                            self.nms_threshold, 0.5)

            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]

                # 转换回原始图像坐标
                x1 = round(box[0] * scale)
                y1 = round(box[1] * scale)
                x2 = round((box[0] + box[2]) * scale)
                y2 = round((box[1] + box[3]) * scale)

                # 确保坐标在图像范围内
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_ids[index],
                    'class_name': self.class_names[class_ids[index]],
                    'confidence': scores[index],
                    'color': self.colors[class_ids[index]]
                })

        return detections

    def process_detections_improved(self, image):
        """
        改进的检测处理流程
        """
        # 方法1: 使用自定义预处理和后处理
        try:
            # 自定义预处理
            blob, scale, original_shape = self.custom_preprocess(image)

            # 使用PyTorch推理 (需要将numpy转换为tensor)
            import torch
            blob_tensor = torch.from_numpy(blob).to(self.model.model.device)

            # 推理
            with torch.no_grad():
                outputs = self.model.model(blob_tensor)

            # 自定义后处理
            detections = self.custom_postprocess(
                outputs[0].cpu().numpy(), scale, original_shape
            )

            return detections

        except Exception as e:
            print(f"自定义处理失败，使用默认方法: {e}")
            # 方法2: 使用Ultralytics默认流程但调整参数
            results = self.model(image, verbose=False, conf=self.confidence_threshold,
                                 iou=self.nms_threshold)
            return self.process_ultralytics_results(results)

    def process_ultralytics_results(self, results):
        """处理Ultralytics结果"""
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

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': confidence,
                        'color': self.colors[class_id]
                    })

        return detections

    def draw_detections_on_canvas(self, detections):
        """在画布上绘制检测结果"""
        self.canvas.delete("all")

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            hex_color = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))

            # 绘制边界框
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=hex_color,
                width=3
            )

            # 绘制标签
            label = f"{det['class_name']} {det['confidence']:.2f}"
            self.canvas.create_text(
                x1, y1 - 5,
                text=label,
                fill=hex_color,
                anchor='sw',
                font=('Arial', 12, 'bold')
            )

        # 显示信息
        info_text = f"FPS: {self.fps:.1f} | 检测目标: {len(detections)} 个"
        self.canvas.create_text(
            10, 10,
            text=info_text,
            fill='yellow',
            anchor='nw',
            font=('Arial', 14, 'bold')
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
                    # 使用改进的检测方法
                    self.current_detections = self.process_detections_improved(frame)

                    # 更新FPS
                    self.update_fps()

                    # 绘制检测结果
                    self.draw_detections_on_canvas(self.current_detections)

                    # 打印信息
                    if self.frame_count == 0 and len(self.current_detections) > 0:
                        print(f"FPS: {self.fps:.1f} | 检测目标: {len(self.current_detections)} 个")
                        for det in self.current_detections[:3]:  # 只显示前3个
                            print(f"  - {det['class_name']}: {det['confidence']:.2f}")

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

        print("开始改进版目标检测...")

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


# 简单版本 - 直接使用Ultralytics但调整参数
class SimpleImprovedDetector:
    """简单改进版本"""

    def __init__(self, region=(0, 0, 1920, 1080), model_path='yolov8n.pt',
                 confidence_threshold=0.25, nms_threshold=0.45):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.camera = dxcam.create(region=region)
        self.camera.start(target_fps=30, video_mode=True)

        self.running = False
        print("简单改进版检测器已初始化")

    def run(self):
        """运行检测"""
        self.running = True
        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    # 关键改进：调整置信度和NMS阈值
                    results = self.model(frame, verbose=False,
                                         conf=self.confidence_threshold,
                                         iou=self.nms_threshold)

                    # 处理结果
                    detections = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                if box.conf.item() >= self.confidence_threshold:
                                    detections.append({
                                        'class': self.model.names[int(box.cls.item())],
                                        'confidence': box.conf.item()
                                    })

                    # 输出结果
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


def main():
    # 方法1: 完整改进版本
    detector = ImprovedDirectXOverlayDetector(
        region=(0, 0, 1920, 1080),
        model_path='../30980-02/chp9/yolov8n.pt',
        confidence_threshold=0.25,  # 降低置信度阈值
        nms_threshold=0.45  # 调整NMS阈值
    )
    detector.start_detection()

    # 方法2: 简单改进版本
    # detector = SimpleImprovedDetector(
    #     region=(0, 0, 1920, 1080),
    #     model_path='../30980-02/chp9/yolov8n.pt',
    #     confidence_threshold=0.25,
    #     nms_threshold=0.45
    # )
    # detector.run()


if __name__ == "__main__":
    main()