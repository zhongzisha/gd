
import sys,os,glob,shutil

sys.path.insert(0, 'F:/gd/yoloV5/')

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
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, \
    compute_offsets, save_predictions_to_envi_xml, LoadImages
from utils.general import xyxy2xywh, xywh2xyxy, box_iou
from numba import jit
"""
从*_gt_5.xml标注文件中提取第5类标签
提取导线图片做导线分割
"""


def load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info):
    print(gt_txt_filename)
    print(gt_xml_filename)
    # 加载gt，分两部分，一部分是txt格式的。一部分是esri xml格式的
    gt_boxes1, gt_labels1 = load_gt_from_txt(gt_txt_filename)
    gt_boxes2, gt_labels2 = load_gt_from_esri_xml(gt_xml_filename,
                                                  gdal_trans_info=gdal_trans_info)
    gt_boxes = gt_boxes1 + gt_boxes2
    gt_labels = gt_labels1 + gt_labels2
    all_boxes = np.concatenate([np.array(gt_boxes, dtype=np.float32).reshape(-1, 4),
                                np.array(gt_labels, dtype=np.float32).reshape(-1, 1)], axis=1)
    print('all_boxes')
    print(all_boxes)

    # 每个类进行nms
    tmp_boxes = []
    tmp_labels = []
    for label in [1, 2, 3, 4, 5]:
        idx = np.where(all_boxes[:, 4] == label)[0]
        if len(idx) > 0:
            boxes_thisclass = all_boxes[idx, :4]
            labels_thisclass = all_boxes[idx, 4]
            if label < 5:  # 这里对导线框不进行nms
                dets = np.concatenate([boxes_thisclass.astype(np.float32),
                                       0.99 * np.ones_like(idx, dtype=np.float32).reshape([-1, 1])], axis=1)
                keep = py_cpu_nms(dets, thresh=0.5)
                tmp_boxes.append(boxes_thisclass[keep])
                tmp_labels.append(labels_thisclass[keep])
            else:
                tmp_boxes.append(boxes_thisclass)
                tmp_labels.append(labels_thisclass)

    if len(tmp_boxes) > 0:
        gt_boxes = np.concatenate(tmp_boxes)
        gt_labels = np.concatenate(tmp_labels)
    else:
        gt_boxes = []
        gt_labels = []

    return gt_boxes, gt_labels


# 这个函数有内存问题，图片太大内存不够
def main_bak(subset='train'):
    source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
    gt_dir = 'E:/gddata/aerial'    # *_gt_5.xml保存在这个目录下了
    save_root = 'E:/line_seg_gt'
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

        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform)
        print(len(gt_boxes), len(gt_labels))

        if len(gt_boxes) == 0:
            continue

        # 首先根据标注生成mask图像
        print('generate mask ...')
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        if True:
            for box, label in zip(gt_boxes, gt_labels):
                if label >=2:
                    xmin, ymin, xmax, ymax = [int(x) for x in box]
                    mask[ymin:ymax, xmin:xmax] = 1
            # mask_savefilename = save_root+"/"+file_prefix+".png"
            # cv2.imwrite(mask_savefilename, mask)
            # cv2.imencode('.png', mask*255)[1].tofile(mask_savefilename)

        print('extract patches ...')
        # 根据gt_boxes中的框的中心点，随机采样mask图像，得到对应的xmin, ymin, xmax, ymax
        # 然后根据这个坐标得到图像和segmap
        for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            if label != 5:
                continue
            xmin, ymin, xmax, ymax = box
            xc = (xmin + xmax) // 2
            yc = (ymin + ymax) // 2
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
            # print(j, xmin1, ymin1, width, height)
            cutout = []
            for bi in range(3):
                band = ds.GetRasterBand(bi + 1)
                band_data = band.ReadAsArray(xmin1, ymin1, win_xsize=width, win_ysize=height)
                cutout.append(band_data)
            cutout = np.stack(cutout, -1)  # RGB
            im1 = cv2.resize(cutout, (512, 512))  # BGR
            mask1 = mask[ymin1:ymax1, xmin1:xmax1]

            save_prefix = '%03d_%010d'%(ti, j)
            cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1)  # 不能有中文
            cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)

            lines.append('%s\n' % save_prefix)

            if np.random.rand() < 0.01:
                cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                            np.concatenate([im1, 255 * np.stack([mask1, mask1, mask1], axis=2)], axis=1))  # 不能有中文
        del mask
        gc.collect()

    if len(lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(lines)

@jit
def box_iou_np(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


def main(subset='train'):
    source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
    gt_dir = 'E:/gddata/aerial'    # *_gt_5.xml保存在这个目录下了
    save_root = 'E:/line_seg_gt'
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

        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform)
        print(len(gt_boxes), len(gt_labels))

        if len(gt_boxes) == 0:
            continue

        # 首先根据标注生成mask图像，存在内存问题！！！
        print('generate mask ...')
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        if True:
            for box, label in zip(gt_boxes, gt_labels):
                if label >=2:
                    xmin, ymin, xmax, ymax = [int(x) for x in box]
                    mask[ymin:ymax, xmin:xmax] = 255
            mask_savefilename = save_root+"/"+file_prefix+".png"
            # cv2.imwrite(mask_savefilename, mask)
            cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

        del mask
        gc.collect()
        continue

        print('extract patches ...')
        # 根据gt_boxes中的框的中心点，随机采样mask图像，得到对应的xmin, ymin, xmax, ymax
        # 然后根据这个坐标得到图像和segmap
        for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            if label != 5:
                continue
            xmin, ymin, xmax, ymax = box
            xc = (xmin + xmax) // 2
            yc = (ymin + ymax) // 2
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
            # print(j, xmin1, ymin1, width, height)

            # 查找gtboxes里面，与当前框有交集的框
            idx1 = np.where(gt_labels >= 2)[0]
            boxes = gt_boxes[idx1, :]  # 得到框
            ious = box_iou_np(np.array([xmin1, ymin1, xmax1, ymax1], dtype=np.float32).reshape(-1, 4), boxes)
            idx2 = np.where(ious > 1e-8)[1]
            if len(idx2) > 0:
                valid_boxes = boxes[idx2, :]
                valid_boxes[:, [0, 2]] -= xmin1
                valid_boxes[:, [1, 3]] -= ymin1
                # import pdb
                # pdb.set_trace()
                mask1 = np.zeros((height, width), dtype=np.uint8)
                for box in valid_boxes.astype(np.int32):
                    xmin, ymin, xmax, ymax = box
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(xmax, width-1)
                    ymax = min(ymax, height-1)
                    mask1[ymin:ymax, xmin:xmax] = 1

                cutout = []
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(xmin1, ymin1, win_xsize=width, win_ysize=height)
                    cutout.append(band_data)
                im1 = np.stack(cutout, -1)  # RGB
                # im1 = cv2.resize(im1, (512, 512))  # BGR

                assert im1.shape[:2] == mask1.shape[:2]

                save_prefix = '%03d_%010d'%(ti, j)
                cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1)  # 不能有中文
                cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)

                lines.append('%s\n' % save_prefix)

                if np.random.rand() < 0.01:
                    cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                                np.concatenate([im1, 255 * np.stack([mask1, mask1, mask1], axis=2)], axis=1))  # 不能有中文

    if len(lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(lines)


if __name__ == '__main__':
    subset = sys.argv[1]
    main(subset=subset)










