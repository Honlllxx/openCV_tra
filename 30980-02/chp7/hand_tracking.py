import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        # 初始化类的参数
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # 初始化手跟踪模块
        self.mpHands = mp.solutions.hands
        # mediapipe version: 0.8.11
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                         self.detectionCon, self.trackCon)
        # mediapipe version: 0.8.3
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        #                                self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # 跟踪手关节点位置
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    # 对手关节点绘制圆圈
    def findPostion(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            for myHand in self.results.multi_hand_landmarks:
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0

    # 打开摄像机Camera0
    cap = cv2.VideoCapture(0)

    # 实例化类对象
    detector = handDetector()

    while True:
        # 读取摄像机的视频图像
        success, img = cap.read()

        # 跟踪关节点位置
        img = detector.findHands(img)

        # 对手关节点绘制圆圈
        lmList = detector.findPostion(img)

        # 计算实时帧率
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # 显示实时帧率
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)

        # 显示视频图像
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()