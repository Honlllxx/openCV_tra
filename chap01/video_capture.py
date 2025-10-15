import cv2
import os


def face_tracking_with_blur():
    # 创建显示视频的窗口
    cv2.namedWindow('Face Tracking')

    # 打开摄像头
    video_capture = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not video_capture.isOpened():
        print("无法打开摄像头")
        return

    # 确保保存目录存在
    if not os.path.exists('data'):
        os.makedirs('data')

    # 创建视频写入对象
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(
        'data/face_tracking.mp4',
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height)
    )

    # 加载人脸检测模型
    try:
        # 尝试加载人脸检测级联分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except AttributeError:
        print("无法找到人脸检测模型，请确保OpenCV安装正确")
        return

    # 读取第一帧
    success, frame = video_capture.read()

    # 循环处理视频帧
    while success and cv2.waitKey(1) != 27:  # 27是ESC键的ASCII码
        # 对帧进行高斯模糊
        blur_frame = cv2.GaussianBlur(frame, (3, 3), 0)

        # 转换为灰度图用于人脸检测
        gray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 在检测到的人脸周围绘制矩形框
        for (x, y, w, h) in faces:
            cv2.rectangle(blur_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(blur_frame, 'Face', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示处理后的帧
        cv2.imshow('Face Tracking', blur_frame)

        # 将处理后的帧写入视频文件
        video_writer.write(blur_frame)

        # 读取下一帧
        success, frame = video_capture.read()

    # 释放资源
    cv2.destroyWindow('Face Tracking')
    video_capture.release()
    video_writer.release()
    print("程序已退出，视频已保存到data/face_tracking.mp4")


if __name__ == "__main__":
    print("启动人脸追踪程序...")
    print("按ESC键退出程序")
    face_tracking_with_blur()
