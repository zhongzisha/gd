import sys,os
import glob
import shutil

import numpy as np

files = glob.glob('E:/patches/train/pos/*.jpg')
files1 = glob.glob('E:/patches/train1/pos/*.jpg')

set1 = set([f.split(os.sep)[-1] for f in files])
set2 = set([f.split(os.sep)[-1] for f in files1])
set3 = set1.difference(set2)

print(len(set1))
print(len(set2))
print(len(set3))

for f in set3:
    src = 'E:/patches/train/pos/'+f
    dst = 'E:/patches/train1/neg/'+f
    shutil.copy(src, dst)








