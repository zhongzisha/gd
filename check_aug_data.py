
import os
import glob
import cv2
import numpy as np
import shutil


save_root = '/media/ubuntu/Data/gd_1024_aug/train/img_shown/'
if os.path.exists(save_root):
    shutil.rmtree(save_root)
if not os.path.exists(save_root):
    os.makedirs(save_root)

files = glob.glob('/media/ubuntu/Data/gd_1024_aug/train/labels/*.txt')
for index, file in enumerate(files):
    prefix = file.split(os.sep)[-1].replace('.txt', '')
    imgname = '/media/ubuntu/Data/gd_1024_aug/train/images/' + prefix + '.jpg'

    with open(file, 'r') as fp:
        lines = [line.strip().split(' ') for line in fp.readlines()]

    lines = np.array(lines)
    labels = lines[:, 0].tolist()
    boxes = lines[:, 1:].astype(np.float32)

    im = cv2.imread(imgname)
    for box, label in zip(boxes, labels):
        xc, yc, w, h = (box * 1024).astype(np.int32)
        xmin = xc - w//2
        ymin = yc - h//2
        xmax = xc + w//2
        ymax = yc + h//2
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color=(0, 0, 255) if label=='1' else (0, 255, 0))
    cv2.imwrite(save_root + '/' + prefix+ '.jpg', im)
    print(file)

    if index == 100:
        break




