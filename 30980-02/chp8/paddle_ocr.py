from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image, ImageDraw, ImageFont
font=cv2.FONT_HERSHEY_SIMPLEX
import numpy as np

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = 'data/imgs/00056221.jpg'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)



# opencv 图像显示中文
def putText_Chinese(img,strText,pos,color,fontSize):
    fontpath = "data/fonts/simfang.ttf"
    font = ImageFont.truetype(fontpath, fontSize)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos,strText, font=font, fill=color)
    img = np.array(img_pil)
    return img

print('---------------OpenCV 可视化---------------------')
img_path = 'data/sgl_ticket.jpg'
img = cv2.imread(img_path)
cv2.imshow("src", img)
result = ocr.ocr(img_path, cls=True)
# print(result)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)
        print('----------------------------')
        print(line)
        pt1 = ((int)(line[0][0][0]), (int)(line[0][0][1]))
        pt2 = ((int)(line[0][1][0]), (int)(line[0][1][1]))
        pt3 = ((int)(line[0][2][0]), (int)(line[0][2][1]))
        pt4 = ((int)(line[0][3][0]), (int)(line[0][3][1]))
        cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, pt2, pt3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, pt3, pt4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, pt1, pt4, (0, 0, 255), 1, cv2.LINE_AA)
        img = putText_Chinese(img, line[1][0], (pt1[0], pt1[1] - 20), (255, 0, 255), 20)

cv2.imshow("OCR-Result", img)
cv2.imwrite("result.png", img)
cv2.waitKey()
cv2.destroyAllWindows()
