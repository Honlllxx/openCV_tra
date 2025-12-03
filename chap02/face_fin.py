import cv2
import time


def face_tracking():
    # 加载预训练的Haar级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置窗口名称
    window_name = "人脸追踪"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 用于计算帧率
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            # 读取一帧图像
            ret, frame = cap.read()

            # 如果读取失败，退出循环
            if not ret:
                print("无法接收帧（摄像头已断开或视频结束）。退出...")
                break

            # 转换为灰度图像，提高检测效率
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # 图像缩放比例
                minNeighbors=5,  # 每个候选矩形应保留的邻近数量
                minSize=(30, 30)  # 目标的最小尺寸
            )

            # 绘制人脸矩形框
            for (x, y, w, h) in faces:
                # 绘制矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 添加文字标签
                cv2.putText(frame, 'Face', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 计算并显示帧率
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # 显示结果帧
            cv2.imshow(window_name, frame)

            # 按下'q'键退出
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # 释放摄像头资源
        cap.release()
        # 关闭所有窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("启动人脸追踪程序...")
    print("按 'q' 键退出程序")
    face_tracking()
    print("程序已退出")
