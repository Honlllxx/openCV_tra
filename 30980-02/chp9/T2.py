import numpy as np
import cv2
from mss import mss
from mss.tools import to_png
import time
import argparse


class ScreenDetector:
    def __init__(self, monitor_index=1, detect_motion=False, detect_color=None, threshold=30):
        """
        初始化屏幕检测器
        :param monitor_index: 监控的屏幕索引，1为 primary 屏幕
        :param detect_motion: 是否检测运动
        :param detect_color: 检测的颜色范围 (lower, upper)，例如红色 ([0, 120, 70], [10, 255, 255])
        :param threshold: 运动检测的阈值
        """
        self.sct = mss()
        self.monitors = self.sct.monitors
        self.monitor = self.monitors[monitor_index]
        self.detect_motion = detect_motion
        self.detect_color = detect_color
        self.threshold = threshold
        self.prev_frame = None
        self.running = False

    def start_detection(self):
        """开始屏幕检测"""
        self.running = True
        print("开始屏幕检测... 按 'q' 键退出")

        while self.running:
            start_time = time.time()

            # 捕获屏幕
            sct_img = self.sct.grab(self.monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 转换颜色空间

            # 复制原始帧用于显示结果
            result_frame = frame.copy()

            # 运动检测
            if self.detect_motion:
                result_frame = self.detect_motion_in_frame(frame, result_frame)

            # 颜色检测
            if self.detect_color:
                result_frame = self.detect_color_in_frame(frame, result_frame)

            # 显示帧率
            fps = 1 / (time.time() - start_time)
            cv2.putText(result_frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示结果
            cv2.imshow("Screen Detection", result_frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_detection()

    def detect_motion_in_frame(self, frame, result_frame):
        """在帧中检测运动"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 初始化前一帧
        if self.prev_frame is None:
            self.prev_frame = gray
            return result_frame

        # 计算当前帧与前一帧的差异
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 找到轮廓
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓
        for c in contours:
            # 忽略小轮廓，减少噪声
            if cv2.contourArea(c) < 500:
                continue

            # 计算轮廓的边界框
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_frame, "Motion Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 更新前一帧
        self.prev_frame = gray
        return result_frame

    def detect_color_in_frame(self, frame, result_frame):
        """在帧中检测特定颜色"""
        lower, upper = self.detect_color

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建颜色掩码
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 找到轮廓
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓
        for c in contours:
            if cv2.contourArea(c) < 300:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result_frame, "Color Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return result_frame

    def stop_detection(self):
        """停止屏幕检测"""
        self.running = False
        cv2.destroyAllWindows()
        print("屏幕检测已停止")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='实时屏幕内容检测程序')
    parser.add_argument('--motion', action='store_true', help='启用运动检测')
    parser.add_argument('--color', nargs=6, type=int, help='检测特定颜色，提供HSV范围值: 低H 低S 低V 高H 高S 高V')
    parser.add_argument('--threshold', type=int, default=30, help='运动检测阈值')
    parser.add_argument('--monitor', type=int, default=1, help='监控的屏幕索引')

    args = parser.parse_args()

    # 处理颜色参数
    detect_color = None
    if args.color:
        detect_color = (args.color[:3], args.color[3:])
        print(f"启用颜色检测，范围: {detect_color}")

    # 创建并启动检测器
    detector = ScreenDetector(
        monitor_index=args.monitor,
        detect_motion=args.motion,
        detect_color=detect_color,
        threshold=args.threshold
    )

    try:
        detector.start_detection()
    except KeyboardInterrupt:
        detector.stop_detection()