import shutil
import os

data_root = '/home/ubuntu/Downloads/Annotations_val/'
for root, dirs, files in os.walk(data_root, topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        if os.path.isfile(filename) and name[-4:] == '.xml':
            print(filename)
            if not os.path.exists(data_root + "/" + filename.split(os.sep)[-1]):
                shutil.move(filename, data_root)