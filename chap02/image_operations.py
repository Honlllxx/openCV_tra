import cv2
img=cv2.imread('data/image_read.jpg')
red = img[0,0,1]
print(red)

print(img.item(0,0,1))

img.itemset((0,0,1),100)

print(img.item(0,0,1))

print(img[0,0,1])

