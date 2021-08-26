

import sys,os,glob,shutil
import numpy as np

data_root = sys.argv[1]
save_root = sys.argv[2]


if not os.path.exists(save_root):
    os.makedirs(os.path.join(save_root, 'train'))
    os.makedirs(os.path.join(save_root, 'val'))


dirs = os.listdir(data_root)

for d in dirs:
    if os.path.isdir(d):
        print(d)
    print(d)
    label_name = d.split(os.sep)[-1]
    print(label_name)

    save_dir = os.path.join(save_root, 'train', label_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_root, 'val', label_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = glob.glob(os.path.join(data_root, d, '*.jpg'))
    np.random.shuffle(files)

    train_count = int(np.ceil(len(files) * 0.8))
    for i in range(train_count):
        filepath = files[i]
        filename = filepath.split(os.sep)[-1]
        save_filepath = os.path.join(save_root, 'train', label_name, filename)
        print(i, filepath, save_filepath)
        shutil.copyfile(filepath, save_filepath)

    for i in range(train_count, len(files)):
        filepath = files[i]
        filename = filepath.split(os.sep)[-1]
        save_filepath = os.path.join(save_root, 'val', label_name, filename)
        print(i, filepath, save_filepath)
        shutil.copyfile(filepath, save_filepath)
















