"""
将提取的杆塔正样本进行通道shuffle增强
看看能不能提升正样本召回率
"""

import sys,os,glob,shutil,cv2

src_pos_dir = r'E:\ganta_patch_classification\train1\pos'
dst_pos_dir = r'E:\ganta_patch_classification\train2\pos'

filenames = glob.glob(os.path.join(src_pos_dir, '*.jpg'))
for filename in filenames:
    fileprefix = filename.split(os.sep)[-1]
    im = cv2.imread(filename)
    cv2.imwrite(os.path.join(dst_pos_dir, fileprefix.replace('.jpg', '_012.jpg')), im)
    cv2.imwrite(os.path.join(dst_pos_dir, fileprefix.replace('.jpg', '_021.jpg')), im[:, :, [0, 2, 1]])
    cv2.imwrite(os.path.join(dst_pos_dir, fileprefix.replace('.jpg', '_210.jpg')), im[:, :, [2, 1, 0]])
    cv2.imwrite(os.path.join(dst_pos_dir, fileprefix.replace('.jpg', '_201.jpg')), im[:, :, [2, 0, 1]])
    cv2.imwrite(os.path.join(dst_pos_dir, fileprefix.replace('.jpg', '_102.jpg')), im[:, :, [1, 0, 2]])
    cv2.imwrite(os.path.join(dst_pos_dir, fileprefix.replace('.jpg', '_120.jpg')), im[:, :, [1, 2, 0]])
    print(filename)