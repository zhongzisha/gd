import sys,os,glob
import cv2
import numpy as np

save_dir="H:/tmp/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
im = cv2.imread('images/001_0000000462.jpg')
cv2.imshow("Source", im)

for step in range(1, 3):
    r = np.random.randint(170, 220)
    g = r - np.random.randint(1, 5)
    b = g - np.random.randint(1, 5)
    line_width = np.random.randint(1, 3)

    H, W = im.shape[:2]
    y = np.random.randint(0, H-1, size=2)
    x = np.random.randint(0, W-1, size=2)
    cv2.line(im, (x[0], y[0]), (x[1], y[1]), color=(b, g, r), thickness=line_width)

cv2.imshow("lined", im)
cv2.imwrite("%s/lined.png"%save_dir, im)

gauss_kernel = cv2.getGaussianKernel(3, 1)
print('gauss_kernel', gauss_kernel)

for ksize in [3, 5, 7, 9]:
    ksize = int(ksize)
    for sigma in range(1, ksize):

        im1 = cv2.GaussianBlur(im.copy(), (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        # cv2.imshow("gauss_blur", im1)
        cv2.imwrite("%s/gauss_blur_ksize=%d_sigma=%d.png" % (save_dir, ksize, sigma), im1)

    im2 = cv2.medianBlur(im.copy(), ksize)
    # cv2.imshow("median_blur")
    cv2.imwrite("%s/median_blur_ksize=%d.png"%(save_dir, ksize), im2)

# cv2.waitKey()