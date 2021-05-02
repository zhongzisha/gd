import sys,os,glob
import cv2
import numpy as np
import math

images_root = sys.argv[1]
save_root = sys.argv[2]

if not os.path.exists(save_root):
    os.makedirs(save_root)

filenames = glob.glob(images_root + '/*.jpg')
for filename in filenames:
    file_prefix = filename.split(os.sep)[-1].replace('.jpg','')
    src = cv2.imread(filename)
    dst = cv2.Canny(src, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi/180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    im1 = np.concatenate([src, cdst, cdstP], axis=1)
    cv2.imwrite(os.path.join(save_root, file_prefix + '.jpg'), im1)
    print(file_prefix + ' done')
    # cv2.imshow("Source", src)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    #
    # cv2.waitKey()

