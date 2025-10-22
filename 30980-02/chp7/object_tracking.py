import cv2
import sys
import cv2.legacy

if __name__ == '__main__':

    # OpenCV提供追踪器列表.
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[6]

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()
    if tracker_type == "MOSSE":
        tracker = cv2.legacy.TrackerMOSSE_create()

    # 读取视频
    video = cv2.VideoCapture("data/Biker.avi")

    # 判断视频是否可以打开
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # 读取第一帧.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # 默认的初始化候选框位置
    bbox = (262, 94, 16, 26)

    # 或者由人工设定，用鼠标绘制
    bbox = cv2.selectROI(frame, False)

    # 由视频第一帧以及模板矩形框来初始化追踪器
    ok = tracker.init(frame, bbox)
    i = 0
    while True:
        # 读取下一帧
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # 更新追踪器
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 追踪结果可视化
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 显示追踪器类型
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # 显示FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # 显示结果
        cv2.imshow("Tracking", frame)

        cv2.imwrite("data/Biker/result/"+ str(int(i)) +".jpg", frame)
        i = i+1

        # 按ESC键退出
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
