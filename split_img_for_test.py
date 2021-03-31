import sys,os,glob
import numpy as np
import cv2
import copy


def savepatches(img, subimgname, left, up, subsize):
    subimg = copy.deepcopy(img[up: (up + subsize), left: (left + subsize)])
    h, w, c = np.shape(subimg)
    outimg = np.zeros((subsize, subsize, 3), dtype=subimg.dtype)
    outimg[0:h, 0:w, :] = subimg
    cv2.imwrite(subimgname, outimg)


def split():
    subsize = 1024
    gap = 128
    slide = subsize - gap
    img_path = ''
    name = ''
    save_img_path = ''
    img = cv2.imread(os.path.join(img_path, name))

    # pad 128
    img = np.pad(img, ((128, 128), (128, 128), (0, 0)))

    outbasename = name + '__'
    width = np.shape(img)[1]
    height = np.shape(img)[0]

    left, up = 0, 0
    while left < width:
        if left + subsize >= width:
            left = max(width - subsize, 0)
        up = 0
        while up < height:
            if up + subsize >= height:
                up = max(height - subsize, 0)
            right = min(left + subsize, width - 1)
            down = min(up + subsize, height - 1)
            subimgname = outbasename + str(left) + '___' + str(up)

            savepatches(img, os.path.join(save_img_path, subimgname + '.png'),
                        left, up, subsize=subsize)

            if up + subsize >= height:
                break
            else:
                up = up + slide
        if left + subsize >= width:
            break
        else:
            left = left + slide







