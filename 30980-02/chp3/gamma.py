import cv2 as cv
import numpy as np
gamma=0.54
img = cv.imread('data/0.tif', flags=0)
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    if i < 30:
        lookUpTable[0, i] = 0
    if i > 61:
        lookUpTable[0, i] = 255
    lookUpTable[0, i] = int((i - 30) * 255 / 31)
    #lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
# res = cv.LUT(img_original, lookUpTable)
# cv.imshow("res", res)
# cv.waitKey(0)

inBlack  = np.array([30, 30, 30], dtype=np.float32)
inWhite  = np.array([61, 61, 61], dtype=np.float32)
inGamma  = np.array([0.54, 0.54, 0.54], dtype=np.float32)
outBlack = np.array([0, 0, 0], dtype=np.float32)
outWhite = np.array([255, 255, 255], dtype=np.float32)

# img = np.clip( (img - inBlack) / (inWhite - inBlack)*255, 0, 255).astype(np.uint8)
img = cv.LUT(img, lookUpTable)
print (img[1000])
#img = ( img ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
res = cv.LUT(img, lookUpTable)
#img = np.clip( img, 0, 255).astype(np.uint8)

cv.imshow("res", img)
cv.waitKey(0)