import cv2
import mss
import numpy as np
from ultralytics import YOLO
import os


class ScreenDetector:
    """
    屏幕目标检测器

    parameters
    ----------
        screen_resolution : Tuple[int, int]
            屏幕分辨率 (宽度, 高度)
        detection_region : Tuple[float, float]
            检测区域比例 (宽度比例, 高度比例)，(1.0, 1.0)表示全屏
        window_name : str
            显示窗口名称
        exit_key : int
            退出键的ASCII码，默认ESC键(27)
        model_path : str
            YOLO模型路径
        confidence_threshold : float
            置信度阈值，只显示高于此值的检测结果
    """

    def __init__(self, screen_resolution=(1920, 1080), detection_region=(1.0, 1.0),
                 window_name='实时目标检测', exit_key=27,
                 model_path='yolov8n.pt', confidence_threshold=0.5):

        # 初始化YOLO模型
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # 屏幕设置
        self.screen_width, self.screen_height = screen_resolution
        self.window_name = window_name
        self.exit_key = exit_key

        # 计算检测区域
        self.detect_width = int(self.screen_width * detection_region[0])
        self.detect_height = int(self.screen_height * detection_region[1])
        self.detect_left = int((self.screen_width - self.detect_width) / 2)
        self.detect_top = int((self.screen_height - self.detect_height) / 2)

        # 截图区域定义
        self.monitor = {
            'left': self.detect_left,
            'top': self.detect_top,
            'width': self.detect_width,
            'height': self.detect_height
        }

        # 显示窗口大小
        self.display_width = self.detect_width // 2
        self.display_height = self.detect_height // 2

        # 初始化截图工具
        self.sct = mss.mss()

        # 加载COCO类别名称
        self.class_names = self.get_class_names()

        # 为每个类别生成随机颜色
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

        print(f"屏幕分辨率: {screen_resolution}")
        print(f"检测区域: {self.detect_width}x{self.detect_height}")
        print(f"显示窗口: {self.display_width}x{self.display_height}")

    def get_class_names(self):
        """获取类别名称"""
        try:
            # 尝试从模型获取类别名称
            if hasattr(self.model, 'names') and self.model.names:
                return self.model.names
            else:
                # 使用COCO数据集默认类别
                return [
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
        except:
            # 备用类别列表
            return [str(i) for i in range(1000)]  # 通用类别编号

    def capture_screen(self):
        """捕获屏幕截图"""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def draw_detections(self, image, results):
        """在图像上绘制检测结果"""
        result_image = image.copy()
        detection_count = 0

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = box.conf.item()

                    # 过滤低置信度检测
                    if confidence < self.confidence_threshold:
                        continue

                    detection_count += 1

                    # 获取边界框坐标和类别
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls.item())
                    class_name = self.class_names[class_id]

                    # 获取该类别的颜色
                    color = self.colors[class_id]

                    # 绘制边界框
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签背景
                    label = f'{class_name} {confidence:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                                  (x1 + label_size[0], y1), color, -1)

                    # 绘制标签文本
                    cv2.putText(result_image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return result_image, detection_count

    def run_detection(self):
        """运行目标检测"""
        print("开始屏幕目标检测...")
        print("按ESC键退出程序")
        print("按'S'键保存当前帧")

        frame_count = 0
        save_folder = "saved_detections"
        os.makedirs(save_folder, exist_ok=True)

        try:
            while True:
                # 捕获屏幕
                screen_img = self.capture_screen()

                # YOLO目标检测
                results = self.model(screen_img, verbose=False)

                # 绘制检测结果
                result_img, detection_count = self.draw_detections(screen_img, results)

                # 显示检测信息
                info_text = f"检测到目标: {detection_count} 个"
                cv2.putText(result_img, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 调整显示大小
                display_img = cv2.resize(result_img, (self.display_width, self.display_height))

                # 显示结果
                cv2.imshow(self.window_name, display_img)

                # 显示控制台信息（每30帧显示一次）
                if frame_count % 30 == 0:
                    print(f"帧: {frame_count}, 检测目标: {detection_count} 个")

                # 键盘操作
                key = cv2.waitKey(1) & 0xFF
                if key == self.exit_key:  # ESC键退出
                    break
                elif key == ord('s') or key == ord('S'):  # S键保存
                    filename = f"{save_folder}/detection_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, result_img)
                    print(f"已保存: {filename}")

                frame_count += 1

        except KeyboardInterrupt:
            print("检测被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            cv2.destroyAllWindows()
            print("目标检测结束")


def main():
    """主函数"""
    # 创建检测器实例
    detector = ScreenDetector(
        screen_resolution=(1920, 1080),  # 根据你的屏幕分辨率调整
        detection_region=(0.8, 0.8),  # 检测区域比例
        confidence_threshold=0.5,  # 置信度阈值
        model_path='../30980-02/chp9/yolov8n.pt'  # 模型路径
    )

    # 运行检测
    detector.run_detection()


if __name__ == "__main__":
    main()