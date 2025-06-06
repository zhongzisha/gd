import sys,os,glob,shutil

import cv2
from PIL import Image
from osgeo import gdal, osr
from pathlib import Path
from natsort import natsorted
import argparse
import psutil  # 获取可用内存
import numpy as np
import torch
import gc
from yoloV5.myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, \
    compute_offsets, save_predictions_to_envi_xml, LoadImages, box_iou_np, \
    load_gt_polys_from_esri_xml
from yoloV5.utils.general import xyxy2xywh, xywh2xyxy, box_iou
from numba import jit

"""
从*_gt_LineRegion14.xml中提取标注的多边形区域，此区域是存在导线的区域
生成mask，为0的表示不存在导线，可在里面提取背景图片，做导线增强
"""


def add_line(im):
    H, W = im.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for step in range(1, 32):
        r = np.random.randint(170, 220)
        g = r - np.random.randint(1, 5)
        b = g - np.random.randint(1, 5)
        line_width = np.random.randint(1, 3)

        y = np.random.randint(0, H - 1, size=2)
        x = np.random.randint(0, W - 1, size=2)
        cv2.line(im, (x[0], y[0]), (x[1], y[1]), color=(b, g, r), thickness=line_width)

        cv2.line(mask, (x[0], y[0]), (x[1], y[1]), color=(1, 0, 0), thickness=line_width)
    return im, mask


def main(subset='train'):
    source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
    gt_dir = 'E:/gddata/aerial'    # *_gt_5.xml保存在这个目录下了
    save_root = 'E:/line_refining_gt_aug'

    random_count = 1
    if subset == 'train':
        random_count = 256

    images_root = save_root + "/images/%s/" % subset
    labels_root = save_root + "/annotations/%s/" % subset
    images_shown_root = save_root + "/images_shown/%s/" % subset
    if not os.path.exists(images_root):
        os.makedirs(images_root)
    if not os.path.exists(labels_root):
        os.makedirs(labels_root)
    if not os.path.exists(images_shown_root):
        os.makedirs(images_shown_root)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    big_subsize = 10240
    gt_gap = 128

    lines = []

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        print("Driver: {}/{}".format(ds.GetDriver().ShortName,
                                     ds.GetDriver().LongName))
        print("Size is {} x {} x {}".format(ds.RasterXSize,
                                            ds.RasterYSize,
                                            ds.RasterCount))
        print("Projection is {}".format(ds.GetProjection()))
        projection = ds.GetProjection()
        projection_sr = osr.SpatialReference(wkt=projection)
        projection_esri = projection_sr.ExportToWkt(["FORMAT=WKT1_ESRI"])
        geotransform = ds.GetGeoTransform()
        xOrigin = geotransform[0]
        yOrigin = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        orig_height, orig_width = ds.RasterYSize, ds.RasterXSize
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
            print("IsNorth = ({}, {})".format(geotransform[2], geotransform[4]))

        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_LineRegion14.xml')

        gt_polys, gt_labels = load_gt_polys_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform,
                                                          mapcoords2pixelcoords=True)
        print(len(gt_polys), len(gt_labels))

        # 在图片中随机采样一些点
        print('generate random sample points ...')
        offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
        random_indices_y = []
        random_indices_x = []
        for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up
            # sub_width = min(orig_width, big_subsize)
            # sub_height = min(orig_height, big_subsize)
            # if xoffset + sub_width > orig_width:
            #     sub_width = orig_width - xoffset
            # if yoffset + sub_height > orig_height:
            #     sub_height = orig_height - yoffset

            # print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
            img0 = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
            for b in range(3):
                band = ds.GetRasterBand(b + 1)
                img0[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)
            img0_sum = np.sum(img0, axis=2)
            indices_y, indices_x = np.where(img0_sum > 0)
            inds = np.arange(len(indices_x))
            np.random.shuffle(inds)
            count = min(random_count, len(inds))
            random_indices_y.append(indices_y[inds[:count]] + yoffset)
            random_indices_x.append(indices_x[inds[:count]] + xoffset)

            del img0, img0_sum, indices_y, indices_x

        random_indices_y = np.concatenate(random_indices_y).reshape(-1, 1)
        random_indices_x = np.concatenate(random_indices_x).reshape(-1, 1)
        print(random_indices_y.shape, random_indices_x.shape)
        print(random_indices_y[:10])
        print(random_indices_x[:10])

        if len(gt_polys) == 0:
            continue

        # 首先根据标注生成mask图像，存在内存问题！！！
        print('generate mask ...')
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        if True:
            # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
            # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)

            for poly, label in zip(gt_polys, gt_labels):  # poly为nx2的点, numpy.array
                cv2.drawContours(mask, [poly], -1, color=(255, 0, 0), thickness=-1)

            mask_savefilename = save_root+"/"+file_prefix+".png"
            # cv2.imwrite(mask_savefilename, mask)
            if not os.path.exists(mask_savefilename):
                cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

        print('extract patches from random boxes ...')
        if True:

            for j, (xc, yc) in enumerate(zip(random_indices_x, random_indices_y)):  # per item
                xmin1, ymin1 = xc - 256, yc - 256
                xmax1, ymax1 = xc + 256, yc + 256
                if xmin1 < 0:
                    xmin1 = 0
                    xmax1 = 512
                if ymin1 < 0:
                    ymin1 = 0
                    ymax1 = 512
                if xmax1 > orig_width - 1:
                    xmax1 = orig_width - 1
                    xmin1 = orig_width - 1 - 512
                if ymax1 > orig_height - 1:
                    ymax1 = orig_height - 1
                    ymin1 = orig_height - 1 - 512

                xmin1 = int(xmin1)
                xmax1 = int(xmax1)
                ymin1 = int(ymin1)
                ymax1 = int(ymax1)
                width = xmax1 - xmin1
                height = ymax1 - ymin1

                # 查找gtboxes里面，与当前框有交集的框
                mask1 = mask[ymin1:ymax1, xmin1:xmax1]
                if mask1.sum() > 0:
                    continue

                mask1 = np.zeros((height, width), dtype=np.uint8)

                cutout = []
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(xmin1, ymin1, win_xsize=width, win_ysize=height)
                    cutout.append(band_data)
                im1 = np.stack(cutout, -1)  # RGB
                # im1 = cv2.resize(im1, (512, 512))  # BGR

                assert im1.shape[:2] == mask1.shape[:2]

                im1, mask1 = add_line(im1)

                save_prefix = '%03d_%010d' % (ti, j)
                cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1)  # 不能有中文
                cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)

                lines.append('%s\n' % save_prefix)

                if np.random.rand() < 0.01:
                    cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                                np.concatenate([im1, 255 * np.stack([mask1, mask1, mask1], axis=2)],
                                               axis=1))  # 不能有中文
        del mask

    if len(lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(lines)


if __name__ == '__main__':
    subset = sys.argv[1]
    main(subset=subset)









