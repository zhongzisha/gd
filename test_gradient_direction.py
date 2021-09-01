import numpy as np
import scipy.signal as sig
from scipy import ndimage
import cv2

# data = np.array([[0, 105, 0], [40, 255, 90], [0, 55, 0]])
# G_x = sig.convolve2d(data, np.array([[-1, 0, 1]]), mode='valid')
# G_y = sig.convolve2d(data, np.array([[-1], [0], [1]]), mode='valid')


im0 = cv2.imread('E:/gd/docs/line/unet/result_img_test/Result_002_0000000098.png')
prob = im0[:, 512:1024, :]
mask = im0[:, 1024:1536,:]
cv2.imshow("prob", prob)
cv2.imshow("mask", mask)
# cv2.waitKey()

# linesP = cv2.HoughLinesP(prob[:, :, 0], 1, np.pi / 180, 150, None, 50, 10)
# prob_show = np.copy(prob)
# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(prob_show, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)
# cv2.imshow("prob_show", prob_show)
# cv2.waitKey()

# mask1 = cv2.medianBlur(mask[:, :, 0], ksize=5)
# cv2.imshow("mask1", mask1)
# cv2.waitKey()
# linesP = cv2.HoughLinesP(mask1, 1, np.pi / 180, 150, None, 50, 10)
# prob_show = np.copy(prob)
# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(prob_show, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)
# cv2.imshow("prob_show", prob_show)
# cv2.waitKey()




# find connected components
labeled, nr_objects = ndimage.label(mask[:, :, 0])
print("Number of objects is {}".format(nr_objects))
# Number of objects is 4
