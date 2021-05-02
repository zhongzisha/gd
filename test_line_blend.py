import sys,os,glob
import cv2
import numpy as np

im = cv2.imread('E:/line_seg_gt/images/train/001_0000000462.jpg')
cv2.imshow("Source", im)
cv2.waitKey()

for step in range(1, 32):
    r = np.random.randint(170, 220)
    g = r - np.random.randint(1, 5)
    b = g - np.random.randint(1, 5)
    line_width = np.random.randint(1, 3)

    H, W = im.shape[:2]
    y = np.random.randint(0, H-1, size=2)
    x = np.random.randint(0, W-1, size=2)
    cv2.line(im, (x[0], y[0]), (x[1], y[1]), color=(b, g, r), thickness=line_width)

cv2.imshow("lined", im)
cv2.waitKey()