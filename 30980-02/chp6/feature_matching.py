import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
img1 = cv.imread('data/box.png',0)          # 模板图像
img2 = cv.imread('data/box_in_scene.png',0) # 待测图像

# 初始化特征检测器
sift = cv.SIFT_create()

# 查找参考图和目标图的SIFT关键点和描述器
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# 显示特征点
kp_img1 = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
cv.imshow("kp1", kp_img1)
kp_img2 = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
cv.imshow("kp2", kp_img2)
cv.waitKey(0)

# 查找两幅图像的最佳匹配点
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# 保存匹配度较高的点
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# 保存10个匹配点
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    result = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    result = cv.polylines(result,[np.int32(dst)],True, (0, 255, 0), 3, cv.LINE_AA)
    cv.imshow("result", result)
    cv.waitKey(0)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv.imshow("feature matching", img3)
cv.waitKey(0)