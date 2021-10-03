import sys,os,glob,shutil
import numpy as np
import cv2

data_root = 'E:/Downloads/BData'

save_dir = os.path.join(data_root, 'images_with_alpha')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for filename in glob.glob(os.path.join(data_root, '01', '*.JPG')):
    prefix = filename.split(os.sep)[-1].replace('.JPG', '')
    im = cv2.imread(filename)
    mask = cv2.imread(os.path.join(data_root, '02', '{}.png'.format(prefix)))
    im_with_alpha = np.concatenate([im, mask[:,:,:1]], axis=2)
    cv2.imwrite(os.path.join(save_dir, prefix + '.png'), im_with_alpha)







