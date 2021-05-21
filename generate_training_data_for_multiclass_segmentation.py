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
import socket
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, \
    compute_offsets, save_predictions_to_envi_xml, LoadImages, box_iou_np, \
    load_gt_polys_from_esri_xml

"""
*_gt_building7.xml,
*_gt_landslide10.xml,
*_gt_tree8.xml,
*_gt_water6.xml,
"""


def main(subset='train'):
    source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
    gt_dir = 'F:/gddata/aerial'    # *_gt_5.xml保存在这个目录下了
    save_root = 'E:/multiclass_segmentation'

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

    gt_postfixes = ['_gt_building7.xml',
                    '_gt_landslide10.xml',
                    '_gt_tree8.xml',
                    '_gt_water6.xml']
    random_gt_ratios = [0.2, 0.8, 0.1, 0.1]
    palette = np.random.randint(
        0, 255, size=(len(gt_postfixes), 3))
    opacity = 0.5

    big_subsize = 5120
    gt_gap = 8

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

        print('loading gt ...')
        all_gt_polys, all_gt_labels = [], []
        for gi, gt_postfix in enumerate(gt_postfixes):
            gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix)

            gt_polys, gt_labels = load_gt_polys_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform,
                                                              mapcoords2pixelcoords=True)
            gt_labels = [gi+1 for _ in range(len(gt_labels))]
            all_gt_polys.append(gt_polys)
            all_gt_labels.append(gt_labels)

            print('class-%d'%(gi+1), len(gt_polys), len(gt_labels))

        # 首先根据标注生成mask图像，存在内存问题！！！
        print('generate mask ...')
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        if True:
            # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
            # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)

            for gt_polys, gt_labels in zip(all_gt_polys, all_gt_labels):
                for poly, label in zip(gt_polys, gt_labels):  # poly为nx2的点, numpy.array
                    cv2.drawContours(mask, [poly], -1, color=(label, label, label), thickness=-1)

            mask_savefilename = save_root + "/" + file_prefix + ".png"
            # cv2.imwrite(mask_savefilename, mask)
            if not os.path.exists(mask_savefilename):
                cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

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

            if False:
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

            # sample points from mask
            mask0 = mask[(yoffset):(yoffset+sub_height), (xoffset):(xoffset+sub_width)]
            for gi, random_gt_ratio in enumerate(random_gt_ratios):
                indices_y, indices_x = np.where(mask0 == (gi+1))
                random_gt_count = int(len(indices_x) * random_gt_ratio)
                if random_gt_count > 0:
                    inds = np.arange(len(indices_x))
                    np.random.shuffle(inds)
                    random_gt_count = min(10, len(inds))
                    random_indices_y.append(indices_y[inds[:random_gt_count]] + yoffset)
                    random_indices_x.append(indices_x[inds[:random_gt_count]] + xoffset)
            del indices_y, indices_x

        random_indices_y = np.concatenate(random_indices_y).reshape(-1, 1)
        random_indices_x = np.concatenate(random_indices_x).reshape(-1, 1)
        print(random_indices_y.shape, random_indices_x.shape)

        if len(gt_polys) == 0:
            continue

        print('extract patches from random boxes ...')
        if True:

            for j, (xc, yc) in enumerate(zip(random_indices_x, random_indices_y)):  # per item
                sub_half_w, sub_half_h = np.random.randint(low=256, high=1024, size=2)

                xmin1, ymin1 = xc - sub_half_w, yc - sub_half_h
                xmax1, ymax1 = xc + sub_half_w, yc + sub_half_h
                if xmin1 < 0:
                    xmin1 = 0
                    xmax1 = 2*sub_half_w
                if ymin1 < 0:
                    ymin1 = 0
                    ymax1 = 2*sub_half_h
                if xmax1 > orig_width - 1:
                    xmax1 = orig_width - 1
                    xmin1 = orig_width - 1 - sub_half_w*2
                if ymax1 > orig_height - 1:
                    ymax1 = orig_height - 1
                    ymin1 = orig_height - 1 - sub_half_h*2

                xmin1 = int(xmin1)
                xmax1 = int(xmax1)
                ymin1 = int(ymin1)
                ymax1 = int(ymax1)
                width = xmax1 - xmin1
                height = ymax1 - ymin1

                # 查找gtboxes里面，与当前框有交集的框
                seg = mask[ymin1:ymax1, xmin1:xmax1]

                cutout = []
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(xmin1, ymin1, win_xsize=width, win_ysize=height)
                    cutout.append(band_data)
                img = np.stack(cutout, -1)  # RGB
                # im1 = cv2.resize(im1, (512, 512))  # BGR

                assert img.shape[:2] == seg.shape[:2]

                save_prefix = '%03d_%010d' % (ti, j)
                cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), img)  # 不能有中文
                cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), seg)

                lines.append('%s\n' % save_prefix)

                if True: #np.random.rand() < 0.01:
                    # cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                    #             np.concatenate([im1, 255 * np.stack([mask1, mask1, mask1], axis=2)],
                    #                            axis=1))  # 不能有中文
                    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                    for label, color in enumerate(palette):
                        color_seg[seg == (label+1), :] = color
                    # convert to BGR
                    color_seg = color_seg[..., ::-1]

                    img = img * (1 - opacity) + color_seg * opacity
                    img = img.astype(np.uint8)
                    cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix), img)
        del mask

    if len(lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(lines)



def main_all(subset='train'):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
        save_root = '/media/ubuntu/Data/multiclass_segmentation'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'    # *_gt_5.xml保存在这个目录下了
        save_root = 'E:/multiclass_segmentation'

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

    gt_postfixes = ['_gt_building7.xml',
                    '_gt_landslide10.xml',
                    '_gt_water6.xml']
                    # '_gt_tree8.xml',
                    # '_gt_flood12.xml']
    random_gt_ratios = [0.2, 0.8, 0.1, 0.1]
    palette = np.random.randint(
        0, 255, size=(len(gt_postfixes), 3))
    opacity = 0.5

    big_subsize = 1024
    gt_gap = 2

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

        print('loading gt ...')
        all_gt_polys, all_gt_labels = [], []
        for gi, gt_postfix in enumerate(gt_postfixes):
            gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix)

            gt_polys, gt_labels = load_gt_polys_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform,
                                                              mapcoords2pixelcoords=True)
            gt_labels = [gi+1 for _ in range(len(gt_labels))]
            all_gt_polys.append(gt_polys)
            all_gt_labels.append(gt_labels)

            print('class-%d'%(gi+1), len(gt_polys), len(gt_labels))

        # 首先根据标注生成mask图像，存在内存问题！！！
        print('generate mask ...')
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        if True:
            # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
            # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)

            for gt_polys, gt_labels in zip(all_gt_polys, all_gt_labels):
                for poly, label in zip(gt_polys, gt_labels):  # poly为nx2的点, numpy.array
                    cv2.drawContours(mask, [poly], -1, color=(label, label, label), thickness=-1)

            mask_savefilename = save_root + "/" + file_prefix + ".png"
            # cv2.imwrite(mask_savefilename, mask)
            if not os.path.exists(mask_savefilename):
                cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

        # 在图片中随机采样一些点
        print('generate samples ...')
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
            img = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
            for b in range(3):
                band = ds.GetRasterBand(b + 1)
                img[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)
            img_sum = np.sum(img, axis=2)
            indices_y, indices_x = np.where(img_sum > 0)
            if len(indices_x) == 0:
                continue

            # sample points from mask
            seg = mask[(yoffset):(yoffset+sub_height), (xoffset):(xoffset+sub_width)]
            seg_count = len(np.where(seg > 0)[0])
            if seg_count < 10:
                continue

            assert img.shape[:2] == seg.shape[:2]

            save_prefix = '%03d_%010d' % (ti, oi)
            cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), img[:, :, ::-1])  # 不能有中文
            cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), seg)

            lines.append('%s\n' % save_prefix)

            if True: #np.random.rand() < 0.01:
                # cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                #             np.concatenate([im1, 255 * np.stack([mask1, mask1, mask1], axis=2)],
                #                            axis=1))  # 不能有中文
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    color_seg[seg == (label+1), :] = color
                # convert to BGR
                color_seg = color_seg[..., ::-1]

                img = img * (1 - opacity) + color_seg * opacity
                img = img.astype(np.uint8)
                cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix), img[:, :, ::-1])
        del mask

    if len(lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(lines)


if __name__ == '__main__':
    subset = sys.argv[1]
    # main(subset=subset)
    main_all(subset)








