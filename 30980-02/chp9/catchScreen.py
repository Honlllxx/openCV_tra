import argparse
import cv2.dnn
import numpy as np
import yaml
import win32gui
import win32con
import win32api
from ultralytics.utils.checks import check_yaml
import mss
import time
import threading
from ctypes import windll

# 确保中文显示正常
windll.shcore.SetProcessDpiAwareness(1)

# 加载类别名称
CLASSES = yaml.safe_load(open(check_yaml('coco128.yaml'), 'r', encoding='utf-8'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 全局变量用于控制线程
running = True


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """绘制边界框和标签"""
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class OverlayWindow:
    """半透明覆盖窗口，用于显示检测框"""

    def __init__(self, monitor_area):
        self.hwnd = None
        self.monitor = monitor_area
        self.left, self.top, self.width, self.height = monitor_area
        self.create_window()

    def create_window(self):
        # 窗口类名
        class_name = "DetectionOverlayClass"
        wc = win32gui.WNDCLASS()
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = class_name
        wc.lpfnWndProc = self.wnd_proc
        wc.hbrBackground = win32gui.GetStockObject(win32con.WHITE_BRUSH)
        wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
        wc.hCursor = win32gui.LoadCursor(None, win32con.IDC_ARROW)

        # 注册窗口类
        win32gui.RegisterClass(wc)

        # 创建窗口
        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST,
            class_name,
            "Detection Overlay",
            win32con.WS_POPUP,
            self.left, self.top, self.width, self.height,
            None, None, wc.hInstance, None
        )

        # 设置窗口透明度
        win32gui.SetLayeredWindowAttributes(self.hwnd, 0, 180, win32con.LWA_ALPHA)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)

    def wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == win32con.WM_PAINT:
            hdc, paint_struct = win32gui.BeginPaint(hwnd)
            win32gui.EndPaint(hwnd, paint_struct)
            return 0
        elif msg == win32con.WM_DESTROY:
            return 0
        else:
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def update_overlay(self, boxes_data):
        """更新覆盖层上的检测框"""
        # 创建空白图像作为画布
        hdc = win32gui.GetDC(self.hwnd)
        mem_dc = win32gui.CreateCompatibleDC(hdc)
        bitmap = win32gui.CreateCompatibleBitmap(hdc, self.width, self.height)
        win32gui.SelectObject(mem_dc, bitmap)

        # 填充透明背景
        brush = win32gui.CreateSolidBrush(0x00FFFFFF)  # 白色背景（会被透明处理）
        win32gui.FillRect(mem_dc, (0, 0, self.width, self.height), brush)
        win32gui.DeleteObject(brush)

        # 转换为OpenCV可操作的图像
        bitmap_info = win32gui.GetObject(bitmap) # 获取位图信息
        bits = win32gui.GetBitmapBits(bitmap, self.width * self.height * 4)
        img = np.frombuffer(bits, dtype=np.uint8).reshape((self.height, self.width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 绘制所有检测框
        for data in boxes_data:
            class_id, confidence, x1, y1, x2, y2 = data
            draw_bounding_box(img, class_id, confidence, x1, y1, x2, y2)

        # 转换回位图并显示
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        win32gui.SetBitmapBits(bitmap, img.tobytes())
        win32gui.BitBlt(hdc, 0, 0, self.width, self.height, mem_dc, 0, 0, win32con.SRCCOPY)

        # 清理资源
        win32gui.DeleteDC(mem_dc)
        win32gui.DeleteObject(bitmap)
        win32gui.ReleaseDC(self.hwnd, hdc)

    def destroy(self):
        """销毁窗口"""
        if self.hwnd:
            win32gui.DestroyWindow(self.hwnd)
            self.hwnd = None


def detection_thread(onnx_model, monitor_area, fps_limit):
    """检测线程：负责处理图像和检测目标"""
    global running

    # 加载模型
    model = cv2.dnn.readNetFromONNX(onnx_model)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 创建覆盖窗口
    overlay = OverlayWindow(monitor_area)

    # 屏幕捕获配置
    with mss.mss() as sct:
        monitor = {"left": monitor_area[0], "top": monitor_area[1],
                   "width": monitor_area[2], "height": monitor_area[3]}
        prev_time = 0

        while running:
            # 控制帧率
            current_time = time.time()
            if current_time - prev_time < 1.0 / fps_limit:
                continue
            prev_time = current_time

            # 捕获屏幕画面
            try:
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                original_image = img.copy()
                height, width = img.shape[:2]
            except Exception as e:
                print(f"捕获图像出错: {e}")
                continue

            # 图像预处理
            length = max(height, width)
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = img
            scale = length / 640
            blob = cv2.dnn.blobFromImage(image, 1 / 255, (640, 640), swapRB=True)

            # 模型推理
            model.setInput(blob)
            outputs = model.forward()
            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            # 解析检测结果
            boxes = []
            scores = []
            class_ids = []
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= 0.25:
                    box = [
                        outputs[0][i][0] - 0.5 * outputs[0][i][2],
                        outputs[0][i][1] - 0.5 * outputs[0][i][3],
                        outputs[0][i][2],
                        outputs[0][i][3]
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            # 非极大值抑制
            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

            # 准备绘制数据
            boxes_data = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                x1 = round(box[0] * scale)
                y1 = round(box[1] * scale)
                x2 = round((box[0] + box[2]) * scale)
                y2 = round((box[1] + box[3]) * scale)
                boxes_data.append((class_ids[index], scores[index], x1, y1, x2, y2))

            # 更新覆盖层
            overlay.update_overlay(boxes_data)

        # 清理
        overlay.destroy()


def screen_detection(onnx_model, monitor_area=None, fps_limit=10):
    """启动检测流程"""
    global running

    # 获取默认监控区域（全屏）
    if not monitor_area:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # 主屏幕
            monitor_area = (monitor["left"], monitor["top"],
                            monitor["width"], monitor["height"])

    # 启动检测线程
    det_thread = threading.Thread(target=detection_thread,
                                  args=(onnx_model, monitor_area, fps_limit))
    det_thread.start()

    # 等待用户输入退出
    print("按 'q' 键退出...")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break

    # 等待线程结束
    det_thread.join()
    cv2.destroyAllWindows()
    print("程序已退出")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.onnx', help='ONNX model path')
    parser.add_argument('--area', type=int, nargs=4, help='Monitor area (left, top, width, height)')
    parser.add_argument('--fps', type=int, default=10, help='Max FPS (default: 10)')
    args = parser.parse_args()

    try:
        screen_detection(args.model, args.area, args.fps)
    except Exception as e:
        print(f"程序出错: {e}")
        running = False