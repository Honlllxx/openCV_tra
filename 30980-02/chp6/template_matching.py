import cv2

# 读取大图
src = cv2.imread('data/image_1.png', 1)
cv2.imshow("ori", src)
imgGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 读取模板图
template = cv2.imread('data/image_1_template.png', 0)

# 模板匹配
# cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED
res = cv2.matchTemplate(imgGray, template, cv2.TM_CCOEFF_NORMED)

# 显示匹配结果
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
top_left=max_loc
h, w = template.shape[:2]
bottom_right=(top_left[0]+w,top_left[1]+h)
cv2.rectangle(src, top_left,bottom_right,(0,255,0),2) #绘制匹配矩形
cv2.imshow("result", src)
cv2.waitKey(0)
