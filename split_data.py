import sys, os, glob,shutil
import time

"""
gdal_retile.py -ps 5120 5120 -overlap 512 -targetDir ./examples2 2.tif
"""


def split(subset='train'):
    """
    python split_data.py /media/ubuntu/Working/rs/guangdong_aerial/aerial2/ /media/ubuntu/Temp/gd/data/aerial2/ 1024 128
    :return:
    """
    import cv2
    import tifffile
    data_root = sys.argv[1]  # "/media/ubuntu/Working/rs/guangdong_aerial/aerial"
    save_root = sys.argv[2]  # "/media/ubuntu/Temp/gd/data/aerial/"
    subsize = int(sys.argv[3])
    gap = int(sys.argv[4])
    tif_files = glob.glob(data_root + '/*.tif')
    lines = []

    save_root = os.path.join(save_root, '%d_%d' % (subsize, gap))

    # 先统计save_root里面有多少个文件夹了
    if os.path.exists(save_root):
        max_id = -1
        for name in os.listdir(save_root):
            if os.path.isdir(save_root + '/' + name):
                try:
                    max_id = max(max_id, int(float(name)))
                except:
                    continue
        start = int(max_id + 1)
    else:
        start = 0

    # import pdb
    # pdb.set_trace()

    for i, tif_file in enumerate(tif_files):
        ii = i + start
        print(i, tif_file)
        save_dir = os.path.join(save_root, '%d' % ii)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = tif_file.split(os.sep)[-1]
        save_filename = "%s/%d.tif" % (data_root, ii)
        cmd_line1 = "mv %s %s" % (tif_file, save_filename)
        os.system(cmd_line1)
        # shutil.move(tif_file, save_filename)
        cmd_line2 = "python gdal_retile.py -ps %d %d -overlap %d -targetDir %s %s" % \
                    (subsize, subsize, gap, save_dir, save_filename)
        os.system(cmd_line2)
        cmd_line3 = "mv %s %s" % (save_filename, tif_file)
        os.system(cmd_line3)
        # shutil.move(save_filename, tif_file)

        # os.system('sleep 10')
        time.sleep(5)
        # 删除那些空的图片，节省空间，RGBA图片转换为RGB图片
        files = glob.glob(save_dir + "/*.tif")
        for file in files:
            im = cv2.imread(file, cv2.IMREAD_UNCHANGED)

            if subset == 'train':
                # 如果是训练集，则把那些全部为空的图片先去掉，
                # 再进行标注，加快标注
                minval = im.min()
                maxval = im.max()
                print(file, minval, maxval)
                if minval == maxval or (minval == 0 and maxval == 0):
                    os.remove(file)
                    continue
            # if im.shape[2] == 4:
            #     im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
            #     cv2.imwrite(file, im)

        lines.append("%d %s\n" % (ii, filename))

        # import pdb
        # pdb.set_trace()

    with open(data_root + '/info.txt', 'w') as fp:
        fp.writelines(lines)


if __name__ == '__main__':
    split()
