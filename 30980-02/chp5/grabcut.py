import numpy as np
import cv2
import sys

COLOR_BLUE = [255, 0, 0]  # 矩形框颜色
COLOR_RED = [0, 0, 255]  # 可能背景绘制颜色
COLOR_GREEN = [0, 255, 0]  # 可能前景绘制颜色
COLOR_BLACK = [0, 0, 0]  # 背景绘制颜色
COLOR_WHITE = [255, 255, 255]  # 前景绘制颜色

DRAW_BG = {'color': COLOR_BLACK, 'val': 0}  # 背景，标记为0
DRAW_FG = {'color': COLOR_WHITE, 'val': 1}  # 前景，标记为1
DRAW_PR_BG = {'color': COLOR_RED, 'val': 2}  # 可能背景，标记为2
DRAW_PR_FG = {'color': COLOR_GREEN, 'val': 3}  # 可能前景，标记为3


class GrabCutApp():
    """
    GrabCutApp 利用grabcut对图像进行前景提取

    USAGE:
        python grabcut.py <image_name>

    README FIRST:
        two windows will show up, one for input and one for output

        at first, in input window, right mouse buttom draw a rectangle around the object with blue.
        then, press 'n' to segment the object (once of a few times). if the effect is not so good,
        please prees key below what you want, the key concept is below:

        key '0' - select area of sure background
        key '1' - select area of sure foreground
        key '2' - select area of probable background
        key '3' - select area of probable foreground

        key 'r' - reset the setup
        key 'n' - segment the object
        key 's' - save the segmented image to "image_name_grabcut.png"
        key 'q' - quit
        key esc - quit
    """

    # 初始化
    def __init__(self, imagename: str) -> None:
        self.img = cv2.imread(imagename)
        if self.img is None:
            print('图像读取失败')
            sys.exit(0)

        self.rect = (0, 0, 1, 1)  # 矩形框初始化
        self.drawing = False
        self.rectangle = False  # 是否开始绘制矩形框
        self.rect_over = False  # 判断矩形是否结束
        self.rect_or_mask = 100  # 矩形框或者mask的种类
        self.value = DRAW_FG
        self.thickness = 3
        self.radius = 5

    # 鼠标回调
    def onmouse(self, event, x, y, flags, param) -> None:
        # 自定义鼠标回调函数
        if event == cv2.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img2.copy()
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y),
                              COLOR_BLUE, self.thickness)
                self.rect = (min(self.ix, x), min(self.iy, y),
                             abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0
        elif event == cv2.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv2.rectangle(self.img, (self.ix, self.iy), (x, y),
                         COLOR_BLUE, self.thickness)
            self.rect = (min(self.ix, x), min(self.iy, y),
                         abs(self.ix - x), abs(self.iy - y))
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

        # 交互操作
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.rect_over == False:
                print("draw object first \n")
            else:
                self.drawing = True
                cv2.circle(self.img, (x, y), self.radius, self.value['color'], -1)
                cv2.circle(self.mask, (x, y), self.radius, self.value['val'], -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.img, (x, y), self.radius, self.value['color'], -1)
                cv2.circle(self.mask, (x, y), self.radius, self.value['val'], -1)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv2.circle(self.img, (x, y), self.radius, self.value['color'], -1)
                cv2.circle(self.mask, (x, y), self.radius, self.value['val'], -1)

    def run(self):
        # 拷贝
        self.img2 = self.img.copy()
        # 初始化一个mask图像
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.output = np.zeros(self.img.shape, np.uint8)

        cv2.namedWindow('output')
        cv2.namedWindow('input')
        cv2.setMouseCallback('input', self.onmouse)
        cv2.moveWindow('input', self.img.shape[1] + 10, 0)

        print('draw a rectangle around the object use right mouse button to draw')

        while (1):
            cv2.imshow('output', self.output)
            cv2.imshow('input', self.img)
            k = cv2.waitKey(1)

            #
            if k == 27 or k == ord('q'):  # esc or 'q' to quit
                break
            elif k == ord('0'):  # BG drawing
                print("mark background regions with left mouse buttom \n")
                self.value = DRAW_BG
            elif k == ord('1'):  # FG drawing
                print("mark foreground regions with left mouse buttom \n")
                self.value = DRAW_FG
            elif k == ord('2'):  # PR_BG drawing
                self.value = DRAW_PR_BG
            elif k == ord('3'):  # PR_FG drawing
                self.value = DRAW_PR_FG
            elif k == ord('s'):  # save image
                bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
                res = np.hstack((self.img2, bar, self.img, bar, self.output))
                cv2.imwrite('grabcut_output_result.png', res)
                print('result saved as grabcut_output_result.png\n')
            elif k == ord('r'):  # restore to original status
                print('reset all settings ...\n')
                self.rect = (0, 0, 1, 1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = DRAW_FG
                self.img = self.img2.copy()

                self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                self.output = np.zeros(self.img.shape, np.uint8)
            elif k == ord('n'):
                print("for finer touchups, mark foreground and background after pressing keys 0-3")
                try:
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    if (self.rect_or_mask == 0):  # original rectangle
                        cv2.grabCut(self.img2, self.mask, self.rect, bgdmodel,
                                    fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif (self.rect_or_mask == 1):  # grabcut after rectangle
                        cv2.grabCut(self.img2, self.mask, self.rect, bgdmodel,
                                    fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
                except:
                    import traceback
                    traceback.print_exc()
            mask2 = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
            cv2.imshow('mask2', mask2)
            self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask2)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = GrabCutApp('data/coins.jpg')
    print(app.__doc__)
    app.run()
