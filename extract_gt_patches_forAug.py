import shutil
import sys, os, glob
import argparse
import cv2
import numpy as np
import torch
from osgeo import gdal, osr
from natsort import natsorted
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, box_iou_np, \
    box_intersection_np, load_gt_polys_from_esri_xml, compute_offsets, alpha_map, elastic_transform_v2
import json
import socket
from PIL import Image, ImageDraw, ImageFilter
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import shapely.geometry as shgeo

"""
从gt中提取图像块，包含gt boxes
"""

def calchalf_iou(poly1, poly2):
    """
        It is not the iou on usual, the iou is the value of intersection over poly1
    """
    inter_poly = poly1.intersection(poly2)
    inter_area = inter_poly.area
    poly1_area = poly1.area
    half_iou = inter_area / poly1_area
    return inter_poly, half_iou


def load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info, valid_labels):
    # print(gt_txt_filename)
    # print(gt_xml_filename)
    # 加载gt，分两部分，一部分是txt格式的。一部分是esri xml格式的
    gt_boxes1, gt_labels1 = load_gt_from_txt(gt_txt_filename)
    gt_boxes2, gt_labels2 = load_gt_from_esri_xml(gt_xml_filename,
                                                  gdal_trans_info=gdal_trans_info)
    gt_boxes = gt_boxes1 + gt_boxes2
    gt_labels = gt_labels1 + gt_labels2

    if len(gt_boxes) == 0:
        return [], []

    all_boxes = np.concatenate([np.array(gt_boxes, dtype=np.float32).reshape(-1, 4),
                                np.array(gt_labels, dtype=np.float32).reshape(-1, 1)], axis=1)
    # print('all_boxes')
    # print(all_boxes)

    # 每个类进行nms
    tmp_boxes = []
    tmp_labels = []
    for label in valid_labels:
        idx = np.where(all_boxes[:, 4] == label)[0]
        if len(idx) > 0:
            boxes_thisclass = all_boxes[idx, :4]
            labels_thisclass = all_boxes[idx, 4]
            dets = np.concatenate([boxes_thisclass.astype(np.float32),
                                   0.99 * np.ones_like(idx, dtype=np.float32).reshape([-1, 1])], axis=1)
            keep = py_cpu_nms(dets, thresh=0.5)
            tmp_boxes.append(boxes_thisclass[keep])
            tmp_labels.append(labels_thisclass[keep])
    gt_boxes = np.concatenate(tmp_boxes)
    gt_labels = np.concatenate(tmp_labels)

    return gt_boxes, gt_labels


def extract_fg_images_bak(subset='train', save_root=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    fg_images_filename = os.path.join(save_root, subset + '_fg_images.npy')
    fg_boxes_filename = os.path.join(save_root, subset + '_fg_boxes.npy')
    if os.path.exists(fg_images_filename) and os.path.exists(fg_boxes_filename):
        return fg_images_filename, fg_boxes_filename

    gt_postfix = '_gt_5.xml'
    valid_labels_set = [1, 2, 3, 4]

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    cache_label_list = [1, 2, 3]
    cache_patches_list = []  # save the extracted gt_img_patches
    cache_boxes_list = []  # save the gt_boxes with the patch

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
        gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix)

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                      valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):  # per item
            if label <= 3:  # 1, 2, 3
                xmin0, ymin0, xmax0, ymax0 = box

                # crop the gt_boxes patch and the gt_boxes within it and save them for further augmentation
                if label in cache_label_list:
                    xc = (xmin0 + xmax0) // 2
                    yc = (ymin0 + ymax0) // 2
                    width, height = xmax0 - xmin0, ymax0 - ymin0
                    width = width * 1.1 + 32
                    height = height * 1.1 + 32
                    xoffset = max(0, xc - width // 2)
                    yoffset = max(0, yc - height // 2)
                    if xoffset + width > orig_width:
                        width = orig_width - xoffset
                    if yoffset + height > orig_height:
                        height = orig_height - yoffset
                    # find the gt_boxes within this box
                    boxes = np.copy(gt_boxes)
                    labels = np.copy(gt_labels)
                    ious = box_iou_np(np.array([xoffset, yoffset, xoffset + width, yoffset + height],
                                               dtype=np.float32).reshape(-1, 4),
                                      boxes)
                    idx2 = np.where(ious > 1e-8)[1]
                    tmp_boxes = []
                    if len(idx2) > 0:
                        valid_boxes = boxes[idx2, :]
                        valid_labels = labels[idx2]
                        valid_boxes[:, [0, 2]] -= xoffset
                        valid_boxes[:, [1, 3]] -= yoffset
                        for box1, label1 in zip(valid_boxes.astype(np.int32), valid_labels):
                            xmin, ymin, xmax, ymax = box1
                            xmin1 = max(1, xmin)
                            ymin1 = max(1, ymin)
                            xmax1 = min(xmax, width - 1)
                            ymax1 = min(ymax, height - 1)
                            # here, check the new gt_box[xmin1, ymin1, xmax1, ymax1]
                            # if the area of new gt_box is less than 0.6 of the original box, then remove this box and
                            # record its position, to put it to zero in the image
                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                            area = (xmax - xmin) * (ymax - ymin)
                            if area1 >= 0.6 * area:
                                tmp_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                    if len(tmp_boxes) > 0:
                        cutout = []
                        for bi in range(3):
                            band = ds.GetRasterBand(bi + 1)
                            band_data = band.ReadAsArray(int(xoffset), int(yoffset),
                                                         win_xsize=int(width),
                                                         win_ysize=int(height))
                            cutout.append(band_data)
                        cutout = np.stack(cutout, -1)  # this is RGB
                        cache_patches_list.append(cutout)
                        cache_boxes_list.append(np.array(tmp_boxes).reshape(-1, 5))

    if len(cache_patches_list) > 0:
        np.save(fg_images_filename, cache_patches_list, allow_pickle=True)
        np.save(fg_boxes_filename, cache_boxes_list, allow_pickle=True)
    return fg_images_filename, fg_boxes_filename


def extract_fg_images_bak2(subset='train', save_root=None, do_rotate=False,
                      update_cache=False):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    fg_images_filename = os.path.join(save_root, subset + '_fg_images.npy')
    fg_boxes_filename = os.path.join(save_root, subset + '_fg_boxes.npy')
    if not update_cache and os.path.exists(fg_images_filename) and os.path.exists(fg_boxes_filename):
        return fg_images_filename, fg_boxes_filename

    if os.path.exists(save_root):
        shutil.rmtree(save_root, ignore_errors=True)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    gt_postfix = '_gt_5.xml'
    valid_labels_set = [1, 2, 3, 4]

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    cache_label_list = [1, 2, 3]
    cache_patches_list = []  # save the extracted gt_img_patches
    cache_boxes_list = []  # save the gt_boxes with the patch

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
        gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix)

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                      valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):  # per item
            if label <= 3:  # 1, 2, 3
                xmin0, ymin0, xmax0, ymax0 = box

                # crop the gt_boxes patch and the gt_boxes within it and save them for further augmentation
                if label in cache_label_list:
                    xc = (xmin0 + xmax0) // 2
                    yc = (ymin0 + ymax0) // 2
                    width, height = xmax0 - xmin0, ymax0 - ymin0
                    length = max(width, height)
                    width = length * 1.1 + 32
                    height = length * 1.1 + 32
                    xoffset = max(1, xc - width // 2)
                    yoffset = max(1, yc - height // 2)
                    if xoffset + width > orig_width:
                        width = orig_width - xoffset - 1
                    if yoffset + height > orig_height:
                        height = orig_height - yoffset - 1
                    # find the gt_boxes within this box
                    boxes = np.copy(gt_boxes)
                    labels = np.copy(gt_labels)
                    ious = box_iou_np(np.array([xoffset, yoffset, xoffset + width, yoffset + height],
                                               dtype=np.float32).reshape(-1, 4),
                                      boxes)
                    idx2 = np.where(ious > 1e-8)[1]
                    tmp_boxes = []
                    if len(idx2) > 0:
                        valid_boxes = boxes[idx2, :]
                        valid_labels = labels[idx2]
                        valid_boxes[:, [0, 2]] -= xoffset
                        valid_boxes[:, [1, 3]] -= yoffset
                        for box1, label1 in zip(valid_boxes.astype(np.int32), valid_labels):
                            xmin, ymin, xmax, ymax = box1
                            xmin1 = max(1, xmin)
                            ymin1 = max(1, ymin)
                            xmax1 = min(xmax, width - 1)
                            ymax1 = min(ymax, height - 1)
                            # here, check the new gt_box[xmin1, ymin1, xmax1, ymax1]
                            # if the area of new gt_box is less than 0.6 of the original box, then remove this box and
                            # record its position, to put it to zero in the image
                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                            area = (xmax - xmin) * (ymax - ymin)
                            if area1 >= 0.6 * area:
                                tmp_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                    if len(tmp_boxes) > 0:
                        cutout = []
                        for bi in range(3):
                            band = ds.GetRasterBand(bi + 1)
                            band_data = band.ReadAsArray(int(xoffset), int(yoffset),
                                                         win_xsize=int(width),
                                                         win_ysize=int(height))
                            cutout.append(band_data)
                        cutout = np.stack(cutout, -1)  # this is RGB
                        cache_patches_list.append(cutout)
                        cache_boxes_list.append(np.array(tmp_boxes).reshape(-1, 5))

    if len(cache_patches_list) > 0:
        np.save(fg_images_filename, cache_patches_list, allow_pickle=True)
        np.save(fg_boxes_filename, cache_boxes_list, allow_pickle=True)
    return fg_images_filename, fg_boxes_filename


def check_dataset(subset='train'):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    info_filename = os.path.join(gt_dir, '{}_infos.csv'.format(subset))
    # if os.path.exists(info_filename):
    #     return -1

    gt_postfix = '_gt_5.xml'
    valid_labels_set = [1, 2, 3, 4]
    save_img = True

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    cache_label_list = [1, 2, 3]
    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

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
        print("Height = {}, Width = {}".format(orig_height, orig_width))
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
            print("IsNorth = ({}, {})".format(geotransform[2], geotransform[4]))

        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix)

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                      valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)

        gt_counts = {}
        for label in valid_labels_set:
            inds = np.where(gt_labels == label)[0]
            print("Class {}: {}".format(label, len(inds)))
            gt_counts[label] = len(inds)
        print('gt_counts', gt_counts)
        lines.append("{},{},{},{},{},{},{},{},{}\n".format(
            ti, orig_width, orig_height, pixelWidth, pixelHeight,
            gt_counts[1], gt_counts[2], gt_counts[3], gt_counts[4]
        ))

    if len(lines) > 0:
        with open(info_filename, 'w', encoding='utf-8-sig') as fp:
            fp.writelines(lines)


def check_fg_images(subset='train', save_root=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    save_dir = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    save_txt_path = '%s/labels/' % save_dir
    for p in [save_img_path, save_txt_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    list_lines = []
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["1", "2", "3", "4"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    gt_postfix = '_gt_5.xml'
    valid_labels_set = [1, 2, 3, 4]
    save_img = True

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    cache_label_list = [1, 2, 3]
    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

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
        gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix)

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                      valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):  # per item
            if label <= 3:  # 1, 2, 3
                xmin0, ymin0, xmax0, ymax0 = box

                # crop the gt_boxes patch and the gt_boxes within it and save them for further augmentation
                if label in cache_label_list:
                    xc = (xmin0 + xmax0) // 2
                    yc = (ymin0 + ymax0) // 2
                    width, height = xmax0 - xmin0, ymax0 - ymin0
                    length = max(width, height)
                    width = length * 1.1 + 32
                    height = length * 1.1 + 32
                    xoffset = max(1, xc - width // 2)
                    yoffset = max(1, yc - height // 2)
                    if xoffset + width > orig_width:
                        width = orig_width - xoffset - 1
                    if yoffset + height > orig_height:
                        height = orig_height - yoffset - 1
                    # find the gt_boxes within this box
                    boxes = np.copy(gt_boxes)
                    labels = np.copy(gt_labels)
                    ious = box_iou_np(np.array([xoffset, yoffset, xoffset + width, yoffset + height],
                                               dtype=np.float32).reshape(-1, 4),
                                      boxes)
                    idx2 = np.where(ious > 1e-8)[1]
                    tmp_boxes = []
                    if len(idx2) > 0:
                        valid_boxes = boxes[idx2, :]
                        valid_labels = labels[idx2]
                        valid_boxes[:, [0, 2]] -= xoffset
                        valid_boxes[:, [1, 3]] -= yoffset
                        for box1, label1 in zip(valid_boxes.astype(np.int32), valid_labels):
                            xmin, ymin, xmax, ymax = box1
                            xmin1 = max(1, xmin)
                            ymin1 = max(1, ymin)
                            xmax1 = min(xmax, width - 1)
                            ymax1 = min(ymax, height - 1)
                            # here, check the new gt_box[xmin1, ymin1, xmax1, ymax1]
                            # if the area of new gt_box is less than 0.6 of the original box, then remove this box and
                            # record its position, to put it to zero in the image
                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                            area = (xmax - xmin) * (ymax - ymin)
                            if area1 >= 0.6 * area:
                                tmp_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                    if len(tmp_boxes) > 0:
                        cutout = []
                        for bi in range(3):
                            band = ds.GetRasterBand(bi + 1)
                            band_data = band.ReadAsArray(int(xoffset), int(yoffset),
                                                         win_xsize=int(width),
                                                         win_ysize=int(height))
                            cutout.append(band_data)
                        cutout = np.stack(cutout, -1)  # this is RGB
                        sub_w = width
                        sub_h = height

                        save_prefix = '%d_%d' % (ti, j)

                        # save image
                        # for coco format
                        single_image = {}
                        single_image['file_name'] = save_prefix + '.jpg'
                        single_image['id'] = image_id
                        single_image['width'] = sub_w
                        single_image['height'] = sub_h
                        data_dict['images'].append(single_image)

                        # for yolo format
                        cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                        list_lines.append('./images/%s.jpg\n' % save_prefix)

                        valid_lines = []
                        for box2 in tmp_boxes:
                            xmin, ymin, xmax, ymax, label = box2

                            if save_img:
                                cv2.rectangle(cutout, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                              color=colors[label], thickness=2)

                            xc1 = int((xmin + xmax) / 2)
                            yc1 = int((ymin + ymax) / 2)
                            w1 = xmax - xmin
                            h1 = ymax - ymin

                            valid_lines.append(
                                "%d %f %f %f %f\n" % (label - 1, xc1 / sub_w, yc1 / sub_h, w1 / sub_w, h1 / sub_h))

                            # for coco format
                            single_obj = {'area': int(w1 * h1),
                                          'category_id': int(label),
                                          'segmentation': []}
                            single_obj['segmentation'].append(
                                [int(xmin), int(ymin), int(xmax), int(ymin),
                                 int(xmax), int(ymax), int(xmin), int(ymax)]
                            )
                            single_obj['iscrowd'] = 0

                            single_obj['bbox'] = int(xmin), int(ymin), int(w1), int(h1)
                            single_obj['image_id'] = image_id
                            single_obj['id'] = inst_count
                            data_dict['annotations'].append(single_obj)
                            inst_count = inst_count + 1

                        image_id = image_id + 1

                        with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                            fp.writelines(valid_lines)

                        if save_img:
                            cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

    if len(list_lines) > 0:
        with open(save_dir + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_dir + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


def extract_patches_and_boxes(image, boxes, xc, yc, w, h, ti, j):
    # im: RGB HxWx3
    # boxes: nx5    xyxy,label
    # xc, yc
    if len(boxes) == 0:
        return [], []

    H, W = image.shape[:2]
    pad_width = int(2 * xc + 1 - W)
    pad_height = int(2 * yc + 1 - H)
    image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)))

    w1 = w * 1.1 + 16
    h1 = h * 1.1 + 16
    xmin1, xmax1 = xc - w1//2, xc + w1//2
    ymin1, ymax1 = yc - h1//2, yc + h1//2
    xmin1 = int(max(1, xmin1))
    xmax1 = int(min(W-1, xmax1))
    ymin1 = int(max(1, ymin1))
    ymax1 = int(min(H-1, ymax1))
    points = []
    for box in boxes:
        xmin, ymin, xmax, ymax, label = box
        points.append(Keypoint(x=xmin, y=ymin))
        points.append(Keypoint(x=xmax, y=ymin))
        points.append(Keypoint(x=xmax, y=ymax))
        points.append(Keypoint(x=xmin, y=ymax))
    kps = KeypointsOnImage(points, shape=image.shape)

    if False:
        save_filename = 'E:/fg_images_shown/fg_%d_%d_%d.jpg' % (ti, j, 0)
        tmp_img = image[ymin1:ymax1, xmin1:xmax1, :].copy()
        for box in np.array(boxes).reshape((-1, 5)).astype(np.int32):
            x1, y1, x2, y2, label = box
            cv2.rectangle(tmp_img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
            cv2.putText(tmp_img, str(label), (x1, y1), fontFace=1, fontScale=1, color=(0, 255, 255), thickness=1)
        cv2.imwrite(save_filename, tmp_img[:, :, ::-1])

    ims_list = []
    gt_boxes_list = []
    for degree in range(5, 360, 5):
        seq = iaa.Sequential([
            iaa.Affine(rotate=degree, fit_output=False)
        ])

        # Augment keypoints and images.
        image_aug, kps_aug = seq(image=image, keypoints=kps)

        # print coordinates before/after augmentation (see below)
        # use after.x_int and after.y_int to get rounded integer coordinates
        quads = []
        for i in range(len(kps.keypoints) // 4):
            p1 = kps_aug.keypoints[4*i]
            p2 = kps_aug.keypoints[4*i+1]
            p3 = kps_aug.keypoints[4*i+2]
            p4 = kps_aug.keypoints[4*i+3]
            quads.append([[p1.x, p1.y],[p2.x, p2.y],[p3.x, p3.y],[p4.x, p4.y]])
        quads = np.array(quads).reshape((-1, 4, 2)).astype(np.int32)

        imgpoly = shgeo.Polygon([(xmin1, ymin1),
                                 (xmax1, ymin1),
                                 (xmax1, ymax1),
                                 (xmin1, ymax1)])

        valid_boxes = []
        for qi, quad in enumerate(quads):
            tmp_poly = shgeo.Polygon([(quad[0, 0], quad[0, 1]),
                                      (quad[1, 0], quad[1, 1]),
                                      (quad[2, 0], quad[2, 1]),
                                      (quad[3, 0], quad[3, 1])])
            try:
                inter_poly, half_iou = calchalf_iou(tmp_poly, imgpoly)
            except:
                import pdb
                pdb.set_trace()

            if half_iou > 0.5:
                xmin2 = np.min(quad[:, 0])
                ymin2 = np.min(quad[:, 1])
                xmax2 = np.max(quad[:, 0])
                ymax2 = np.max(quad[:, 1])
                xmin1_new = max(xmin2 - xmin1, 1)
                ymin1_new = max(ymin2 - ymin1, 1)
                xmax1_new = min(xmax2 - xmin1, xmax1 - xmin1 - 1)
                ymax1_new = min(ymax2 - ymin1, ymax1 - ymin1 - 1)
                valid_boxes.append([xmin1_new, ymin1_new, xmax1_new, ymax1_new, boxes[qi, 4]])
        if len(valid_boxes) > 0:
            ims_list.append(image_aug[ymin1:ymax1, xmin1:xmax1, :])
            gt_boxes_list.append(np.array(valid_boxes).reshape((-1, 5)))

            if False:
                save_filename = 'E:/fg_images_shown/fg_%d_%d_%d.jpg' %(ti, j, degree)
                tmp_img = image_aug[ymin1:ymax1, xmin1:xmax1, :]
                for box in np.array(valid_boxes).reshape((-1, 5)).astype(np.int32):
                    x1, y1, x2, y2, label = box
                    cv2.rectangle(tmp_img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
                    cv2.putText(tmp_img, str(label), (x1, y1), fontFace=1, fontScale=1, color=(0, 255, 255), thickness=1)
                cv2.imwrite(save_filename, tmp_img[:, :, ::-1])
    return ims_list, gt_boxes_list


def extract_fg_images(subset='train', save_root=None, do_rotate=False, update_cache=False, debug=False):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    fg_images_filename = os.path.join(save_root, subset + '_fg_images.npy')
    fg_boxes_filename = os.path.join(save_root, subset + '_fg_boxes.npy')
    if not update_cache and os.path.exists(fg_images_filename) and os.path.exists(fg_boxes_filename):
        return fg_images_filename, fg_boxes_filename

    # if os.path.exists(save_root):
    #     shutil.rmtree(save_root, ignore_errors=True)
    if os.path.exists(fg_images_filename):
        os.remove(fg_images_filename)
    if os.path.exists(fg_boxes_filename):
        os.remove(fg_boxes_filename)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    gt_postfix = '_gt_5.xml'
    valid_labels_set = [1, 2, 3, 4]

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    cache_label_list = [1, 2, 3]
    cache_patches_list = []  # save the extracted gt_img_patches
    cache_boxes_list = []  # save the gt_boxes with the patch

    if debug:
        lines = {}
        for label in cache_label_list:
            if not os.path.exists('/media/ubuntu/Data/GantaHQ_Aug/%s/c%d' % (subset, label)):
                os.makedirs('/media/ubuntu/Data/GantaHQ_Aug/%s/c%d' % (subset, label))
            lines1 = glob.glob(
                '/media/ubuntu/Data/gd_newAug1_Rot0_4classes/check_fg_images_v1/%s/Ganta_%d/*.jpg' % (subset, label))
            lines1 = [line.split(os.sep)[-1].replace('.jpg', '') for line in lines1]
            print('lines', lines1)
            lines[label] = lines1
        valid_lines = []

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
        gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix)

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                      valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):  # per item
            if label <= 3:  # 1, 2, 3
                xmin0, ymin0, xmax0, ymax0 = box
                width0, height0 = xmax0 - xmin0, ymax0 - ymin0
                if width0 > 800 or height0 > 800 or width0 < 10 or height0 < 10:
                    continue

                # crop the gt_boxes patch and the gt_boxes within it and save them for further augmentation
                if label in cache_label_list:
                    xc = (xmin0 + xmax0) // 2
                    yc = (ymin0 + ymax0) // 2
                    width0, height0 = xmax0 - xmin0, ymax0 - ymin0
                    length = max(width0, height0)
                    if debug:
                        width = length * 1.1 + 32
                        height = length * 1.1 + 32
                    else:
                        width = length * 2 + 32
                        height = length * 2 + 32
                    width0 = width
                    xoffset = max(1, xc - width // 2)
                    yoffset = max(1, yc - height // 2)
                    if xoffset + width > orig_width:
                        width = orig_width - xoffset - 1
                    if yoffset + height > orig_height:
                        height = orig_height - yoffset - 1
                    # find the gt_boxes within this box
                    boxes = np.copy(gt_boxes)
                    labels = np.copy(gt_labels)
                    ious = box_iou_np(np.array([xoffset, yoffset, xoffset + width, yoffset + height],
                                               dtype=np.float32).reshape(-1, 4),
                                      boxes)
                    idx2 = np.where(ious > 1e-8)[1]
                    tmp_boxes = []
                    if len(idx2) > 0:
                        valid_boxes = boxes[idx2, :]
                        valid_labels = labels[idx2]
                        valid_boxes[:, [0, 2]] -= xoffset
                        valid_boxes[:, [1, 3]] -= yoffset
                        for box1, label1 in zip(valid_boxes.astype(np.int32), valid_labels):
                            xmin, ymin, xmax, ymax = box1
                            xmin1 = max(1, xmin)
                            ymin1 = max(1, ymin)
                            xmax1 = min(xmax, width - 1)
                            ymax1 = min(ymax, height - 1)
                            # here, check the new gt_box[xmin1, ymin1, xmax1, ymax1]
                            # if the area of new gt_box is less than 0.6 of the original box, then remove this box and
                            # record its position, to put it to zero in the image
                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                            area = (xmax - xmin) * (ymax - ymin)
                            if area1 >= 0.6 * area:
                                tmp_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                    if len(tmp_boxes) > 0:
                        cutout = []
                        for bi in range(3):
                            band = ds.GetRasterBand(bi + 1)
                            band_data = band.ReadAsArray(int(xoffset), int(yoffset),
                                                         win_xsize=int(width),
                                                         win_ysize=int(height))
                            cutout.append(band_data)
                        cutout = np.stack(cutout, -1)  # this is RGB

                        tmp_patches_list = [cutout]
                        tmp_boxes_list = [np.array(tmp_boxes).reshape(-1, 5)]

                        if do_rotate:
                            print('fg ', ti, j, width0, cutout.shape)
                            patches_list, boxes_list = extract_patches_and_boxes(cutout, np.array(tmp_boxes).reshape(-1, 5),
                                                                                 xc-xoffset, yc-yoffset, width0, width0,
                                                                                 ti, j)
                            print('rotate done')
                            if len(boxes_list) > 0:
                                tmp_patches_list += patches_list
                                tmp_boxes_list += boxes_list

                        if len(tmp_boxes_list) > 0:
                            cache_patches_list += tmp_patches_list
                            cache_boxes_list += tmp_patches_list

                            if debug:
                                if len(lines[label]) > 0:
                                    if '%d_%d' % (ti, j) in lines[label]:
                                        for ii, patch in enumerate(tmp_patches_list):
                                            save_filename = "%s/c%d/%d_%d_%d.png" % (subset, label, ti, j, ii)
                                            cv2.imwrite("/media/ubuntu/Data/GantaHQ_Aug/%s" % (save_filename),
                                                        patch[:, :, ::-1])
                                            valid_lines.append(save_filename+'\n')
    if debug:
        if len(valid_lines) > 0:
            with open("/media/ubuntu/Data/GantaHQ_Aug/%s.txt" % subset, "w") as fp:
                fp.writelines(valid_lines)
        return None, None

    if len(cache_patches_list) > 0:
        np.save(fg_images_filename, np.array(cache_patches_list, dtype=object), allow_pickle=True)
        np.save(fg_boxes_filename, np.array(cache_boxes_list, dtype=object), allow_pickle=True)
    return fg_images_filename, fg_boxes_filename


def extract_bg_images(subset='train', save_root=None, random_count=0, update_cache=False):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    save_dir = '%s/%s_bg_images/' % (save_root, subset)

    if not update_cache and os.path.exists(save_dir) and len(glob.glob(save_dir + '/*.png')) > 0:
        return save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    if random_count > 0:
        # based on the cached image patches
        # first extract the background image
        # then paste the image patch into the bg image
        big_subsize = 10240
        gt_gap = 128
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

            gt_boxes, gt_boxes_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                valid_labels=[1, 2, 3, 4])

            gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_LineRegion14.xml')

            gt_polys, gt_labels = load_gt_polys_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform,
                                                              mapcoords2pixelcoords=True)
            print(len(gt_polys), len(gt_labels))

            # 首先根据标注生成mask图像，存在内存问题！！！
            print('generate mask ...')
            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            if True:
                # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
                # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)

                if len(gt_boxes) > 0:
                    for box in gt_boxes:
                        xmin, ymin, xmax, ymax = box
                        poly = np.array([[xmin, ymin], [xmax, ymin],
                                         [xmax, ymax], [xmin, ymax]], dtype=np.int32).reshape([4, 2])
                        cv2.drawContours(mask, [poly], -1, color=(255, 0, 0), thickness=-1)

                for poly, label in zip(gt_polys, gt_labels):  # poly为nx2的点, numpy.array
                    cv2.drawContours(mask, [poly], -1, color=(255, 0, 0), thickness=-1)

                # mask_savefilename = save_root + "/" + file_prefix + ".png"
                # cv2.imwrite(mask_savefilename, mask)
                # if not os.path.exists(mask_savefilename):
                #     cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

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

            for j, (xc, yc) in enumerate(zip(random_indices_x, random_indices_y)):  # per item

                w = np.random.randint(low=600, high=1024)
                h = np.random.randint(low=600, high=1024)
                xmin1, ymin1 = xc - w / 2, yc - h / 2
                xmax1, ymax1 = xc + w / 2, yc + h / 2
                if xmin1 < 0:
                    xmin1 = 0
                    xmax1 = w
                if ymin1 < 0:
                    ymin1 = 0
                    ymax1 = h
                if xmax1 > orig_width - 1:
                    xmax1 = orig_width - 1
                    xmin1 = orig_width - 1 - w
                if ymax1 > orig_height - 1:
                    ymax1 = orig_height - 1
                    ymin1 = orig_height - 1 - h

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

                cutout = []
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(xmin1, ymin1, win_xsize=width, win_ysize=height)
                    cutout.append(band_data)
                im1 = np.stack(cutout, -1)  # RGB

                cv2.imencode('.png', im1)[1].tofile('%s/bg_%d_%d.png' % (save_dir, ti, j))

            del mask
    return save_dir


def compose_fg_bg_v1(bg, fg_images_list, fg_boxes_list, inds):  # not good
    # bg: HxWx3 RGB
    # fg_ims: list of RGB images [hxwx3]
    # fg_boxes: list of boxes [nx5]

    im = np.copy(bg)
    H, W = im.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    boxes = []
    for ind in inds:
        fg = np.zeros((H, W, 3), dtype=np.uint8)
        fg_im = np.copy(fg_images_list[ind])  # RGB
        fg_boxes = np.copy(fg_boxes_list[ind])  # nx5
        if len(fg_boxes) == 0:
            continue
        h, w = fg_im.shape[:2]
        area = h * w
        is_ok = False
        xc, yc = -1, -1
        for step in range(10):
            mask1 = np.zeros((H, W), dtype=np.uint8)
            xc = np.random.randint(low=int(w // 2 + 1), high=int(W - w // 2 - 1))
            yc = np.random.randint(low=int(h // 2 + 1), high=int(H - h // 2 - 1))
            left = xc - w // 2
            up = yc - h // 2
            mask1[up:(up + h), left:(left + h)] = 1
            area1 = len(np.where(mask & mask1)[0])
            if area1 < 0.4 * area:
                is_ok = True
                break
        if is_ok:
            left = xc - w // 2
            up = yc - h // 2
            alpha = alpha_map(W, H, w, h, xc, yc)
            fg[up:(up + h), left:(left + w), :] = fg_im
            mask[up:(up + h), left:(left + h)] = 1
            im = alpha * fg + (1 - alpha) * im
            im = im.astype(np.uint8)

            fg_boxes[:, [0, 2]] += left
            fg_boxes[:, [1, 3]] += up
            boxes.append(fg_boxes)
    if len(boxes) > 0:
        boxes = np.concatenate(boxes, axis=0)
        return im, boxes  # RGB, nx5
    else:
        return [], []


def compose_fg_bg(bg, fg_images_list, fg_boxes_list, inds):  # not good
    # bg: HxWx3 RGB
    # fg_ims: list of RGB images [hxwx3]
    # fg_boxes: list of boxes [nx5]

    im = Image.fromarray(np.copy(bg))
    W, H = im.size
    mask = np.zeros((H, W), dtype=np.uint8)
    boxes = []
    blur_radius = 10
    for ind in inds:
        fg_im = np.copy(fg_images_list[ind])
        h, w = fg_im.shape[:2]
        if h > H - 32 or w > W - 32:
            continue

        fg_boxes0 = np.copy(fg_boxes_list[ind])  # nx5
        valid_inds = np.where(fg_boxes0[:, 4] >= 2)[0]
        fg_boxes = fg_boxes0[valid_inds, :5]

        if len(fg_boxes) == 0:
            continue
        # print('fg_boxes', fg_boxes)

        fg_boxes_xmin = np.min(fg_boxes[:, 0])
        fg_boxes_ymin = np.min(fg_boxes[:, 1])
        fg_boxes_xmax = np.max(fg_boxes[:, 2])
        fg_boxes_ymax = np.max(fg_boxes[:, 3])

        is_ok = False
        xc, yc = -1, -1
        for step in range(10):
            xc = np.random.randint(low=int(w // 2 + 1), high=int(W - w // 2 - 1))
            yc = np.random.randint(low=int(h // 2 + 1), high=int(H - h // 2 - 1))
            left = xc - w // 2
            up = yc - h // 2
            if len(boxes) == 0:
                is_ok = True
                break
            mask1 = np.zeros((H, W), dtype=np.uint8)
            mask1[up:(up + h), left:(left + w)] = 1
            area1 = len(np.where(mask & mask1)[0])
            if area1 < 10:
                is_ok = True
                break
        if is_ok:
            left = xc - w // 2
            up = yc - h // 2

            fg = np.zeros((H, W, 3), dtype=np.uint8)
            mask1 = np.zeros((H, W), dtype=np.uint8)
            fg[up:(up + h), left:(left + w), :] = fg_im
            mask[up:(up + h), left:(left + w)] = 1
            mask1_im = Image.fromarray(mask1)
            draw1 = ImageDraw.Draw(mask1_im)
            draw1.rectangle((left + fg_boxes_xmin, up + fg_boxes_ymin, left + fg_boxes_xmax, up + fg_boxes_ymax),
                            fill=255)
            mask1_im_blur = mask1_im.filter(ImageFilter.GaussianBlur(blur_radius))
            im.paste(Image.fromarray(fg), (0, 0), mask1_im_blur)

            fg_boxes[:, [0, 2]] += left
            fg_boxes[:, [1, 3]] += up
            boxes.append(fg_boxes)
    if len(boxes) > 0:
        boxes = np.concatenate(boxes, axis=0)
        return np.array(im), boxes  # RGB, nx5
    else:
        return [], []


def compose_fg_bg_images(subset='train', aug_times=1, save_root=None,
                         fg_images_filename=None, fg_boxes_filename=None,
                         bg_images_dir=None):
    save_root = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    save_img_path = '%s/images/' % save_root
    save_img_shown_path = '%s/images_shown/' % save_root
    save_txt_path = '%s/labels/' % save_root
    for p in [save_img_path, save_txt_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    print(fg_images_filename)
    print(fg_images_filename)
    print(bg_images_dir)
    print('compose fg and bg to new train images ...')
    fg_images_list = np.load(fg_images_filename, allow_pickle=True)  # list of RGB images [HxWx3]
    fg_boxes_list = np.load(fg_boxes_filename, allow_pickle=True)  # list of boxes [nx5]
    fg_inds = np.arange(len(fg_images_list))

    list_lines = []
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["1", "2", "3", "4"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

    aug_times = aug_times if subset == 'train' else 1
    bg_filenames = glob.glob(bg_images_dir + '/*.png')

    bins = np.arange(0, 256)
    for aug_time in range(aug_times):
        print('aug_time', aug_time)

        if len(bg_filenames) > 5000:
            bg_indices = np.random.choice(np.arange(len(bg_filenames)), size=5000, replace=False)
        else:
            bg_indices = np.arange(len(bg_filenames))

        for bg_ind in bg_indices:

            bg_filename = bg_filenames[bg_ind]
            file_prefix = bg_filename.split(os.sep)[-1].replace('.png', '')
            bg = cv2.imread(bg_filename)

            bg_shape = bg.shape[:2]
            if min(bg_shape) < 500 or max(bg_shape) > 2048:
                continue

            hist, _ = np.histogram(bg[:, :, 0], bins=bins)
            if (hist[np.argmax(hist)] / np.prod(bg_shape)) > 0.5:
                continue

            print('bg_ind', bg_ind)
            selected_fg_inds = np.random.choice(fg_inds, size=np.random.randint(1, 5), replace=False)
            print('selected_fg_inds', selected_fg_inds)
            im, gt_boxes = compose_fg_bg(bg, fg_images_list, fg_boxes_list, selected_fg_inds)

            save_img = False

            # draw gt boxes
            if len(gt_boxes) > 0:
                print('num_gt_boxes', len(gt_boxes))

                save_prefix = '%s_%d' % (file_prefix, aug_time)
                sub_h, sub_w = im.shape[:2]

                # save image
                # for coco format
                single_image = {}
                single_image['file_name'] = save_prefix + '.jpg'
                single_image['id'] = image_id
                single_image['width'] = sub_w
                single_image['height'] = sub_h
                data_dict['images'].append(single_image)

                # for yolo format
                cv2.imwrite(save_img_path + save_prefix + '.jpg', im[:, :, ::-1])  # RGB --> BGR

                list_lines.append('./images/%s.jpg\n' % save_prefix)

                if np.random.rand() < 0.05:
                    save_img = True

                valid_lines = []
                for box2 in gt_boxes.astype(np.int32):
                    xmin, ymin, xmax, ymax, label = box2

                    if save_img:
                        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color=colors[label],
                                      thickness=3)

                    xc1 = int((xmin + xmax) / 2)
                    yc1 = int((ymin + ymax) / 2)
                    w1 = xmax - xmin
                    h1 = ymax - ymin

                    valid_lines.append(
                        "%d %f %f %f %f\n" % (label - 1, xc1 / sub_w, yc1 / sub_h, w1 / sub_w, h1 / sub_h))

                    # for coco format
                    single_obj = {'area': int(w1 * h1),
                                  'category_id': int(label),
                                  'segmentation': []}
                    single_obj['segmentation'].append(
                        [int(xmin), int(ymin), int(xmax), int(ymin),
                         int(xmax), int(ymax), int(xmin), int(ymax)]
                    )
                    single_obj['iscrowd'] = 0

                    single_obj['bbox'] = int(xmin), int(ymin), int(w1), int(h1)
                    single_obj['image_id'] = image_id
                    single_obj['id'] = inst_count
                    data_dict['annotations'].append(single_obj)
                    inst_count = inst_count + 1

                image_id = image_id + 1

                with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                    fp.writelines(valid_lines)

                if save_img:
                    cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', im[:, :, ::-1])  # RGB --> BGR

            # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
            # im /= 255.0  # 0 - 255 to 0.0 - 1.0
            # ims.append(im)

    if len(list_lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_root + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


def add_line(im, mask, p0s, p1s):
    line_width = np.random.randint(1, 3)
    for p0, p1 in zip(p0s, p1s):

        d = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        if d < 30:
            continue

        r = np.random.randint(160, 220)
        g = r - np.random.randint(1, 10)
        b = g - np.random.randint(1, 10)

        cv2.line(im, p0, p1, color=(r, g, b), thickness=line_width)
        cv2.line(mask, p0, p1, color=(1, 0, 0), thickness=line_width)

    return im, mask


def add_line_to_image(im, crop_width=0, crop_height=0):
    # extract sub image from im and add line to the image
    # return the croppped image and its line mask
    H, W = im.shape[:2]
    # im_sub = im[(H//2-crop_height//2):(H//2-crop_height//2+crop_height),
    #          (W//2-crop_width//2):(W//2-crop_width//2+crop_width), :]

    if crop_width == 0 or crop_height == 0:
        xc = np.random.randint(low=(crop_width + 1) // 2, high=(W - (crop_width + 1) // 2 - 1))
        yc = np.random.randint(low=(crop_height + 1) // 2, high=(H - (crop_height + 1) // 2 - 1))
        im_sub = im[(yc - crop_height // 2):(yc - crop_height // 2 + crop_height),
                 (xc - crop_width // 2):(xc - crop_width // 2 + crop_width),
                 :]
        if len(np.unique(im_sub)) < 50:
            return None, None
        H, W = crop_height, crop_width
    else:
        im_sub = im

    mask = np.zeros((H, W), dtype=np.uint8)
    for step in range(np.random.randint(2, 5)):
        y = np.random.randint(0, H - 1, size=2)
        x = np.random.randint(0, W - 1, size=2)

        x1, x2 = x
        y1, y2 = y
        if abs(x1 - x2) == 1 and (20 < x1 < W - 20):
            expand = np.random.randint(low=10, high=x1-9, size=2)
            x1_l = x1 - expand[0]
            x1_r = x1 + expand[1]
            p0s, p1s = [], []
            if x1_l >= 3:
                p0s.append((x1_l, 0))
                p1s.append((x1_l, H-1))
            p0s.append((x1, 0))
            p1s.append((x1, H-1))
            if x1_r <= W-3:
                p0s.append((x1_r, 0))
                p1s.append((x1_r, H-1))

            im_sub, mask = add_line(im_sub, mask, p0s, p1s)
        elif abs(y1 - y2) == 1 and (20 < y1 < H - 20):
            expand = np.random.randint(low=10, high=y1-9, size=2)
            y1_u = y1 - expand[0]
            y1_b = y1 + expand[1]
            p0s, p1s = [], []
            if y1_u >= 3:
                p0s.append((0, y1_u))
                p1s.append((W - 1, y1_u))
            p0s.append((0, y1))
            p1s.append((W - 1, y1))
            if y1_b <= H - 3:
                p0s.append((0, y1_b))
                p1s.append((W-1, y1_b))

            im_sub, mask = add_line(im_sub, mask, p0s, p1s)
        elif abs(x1 - x2) > 10 and abs(y1 - y2) > 10:
            k = (y1 - y2) / (x2 - x1)
            b = - k * x1 - y1
            # (y1 - y2)x + (x1 - x2)y + x1y2 -y1x2 = 0
            expand = np.random.randint(low=10, high=100, size=2)
            b_up = b - expand[0]
            b_down = b + expand[1]
            if abs(k) > 1:
                # y=3, y=H-3
                p0s = [(int((-3 - bb)/k), 3) for bb in [b, b_up, b_down]]
                p1s = [(int((-H+3 - bb)/k), H-3) for bb in [b, b_up, b_down]]
                im_sub, mask = add_line(im_sub, mask, p0s, p1s)
            elif 1 >= abs(k) > 0.05:
                # x=3, x=W-3
                p0s = [(3, int(-k*3-bb)) for bb in [b, b_up, b_down]]
                p1s = [(W-3, int(-k*(W-3)-bb)) for bb in [b, b_up, b_down]]
                im_sub, mask = add_line(im_sub, mask, p0s, p1s)

    # blur the image
    prob = np.random.rand()
    if prob < 0.5:
        ksize = np.random.choice([3, 5, 7])
        sigmas = np.arange(0.5, ksize, step=0.5)
        im_sub = cv2.GaussianBlur(im_sub, ksize=(ksize, ksize),
                                  sigmaX=np.random.choice(sigmas),
                                  sigmaY=np.random.choice(sigmas))
    elif 0.5 <= prob <= 0.8:
    #     ksize = np.random.choice([3, 5])
    #     im_sub = cv2.medianBlur(im_sub, ksize=ksize)
    # else:
        im_sub_with_mask = np.concatenate([im_sub, mask[:, :, None]], axis=2)
        im_sub_with_mask = elastic_transform_v2(im_sub_with_mask, im_sub.shape[1] * 2,
                                                im_sub.shape[1] * np.random.randint(low=4, high=8) / 100,
                                                im_sub.shape[1] * np.random.randint(low=4, high=8) / 100)
        im_sub, mask = im_sub_with_mask[:, :, :3], im_sub_with_mask[:, :, 3]

    return im_sub, mask


def refine_line_aug(subset='train', aug_times=1, save_root=None,
                    crop_height=512, crop_width=512,
                    fg_images_filename=None, fg_boxes_filename=None,
                    bg_images_dir=None, random_count=1000):
    save_dir = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = '%s/images/' % (save_dir)
    images_shown_root = '%s/images_shown/' % (save_dir)
    labels_root = '%s/annotations/' % (save_dir)
    for p in [images_root, labels_root, images_shown_root]:
        if not os.path.exists(p):
            os.makedirs(p)

    aug_times = aug_times if subset == 'train' else 1
    bg_filenames = glob.glob(bg_images_dir + '/*.png')

    lines = []
    for aug_time in range(aug_times):
        if len(bg_filenames) > random_count:
            bg_indices = np.random.choice(np.arange(len(bg_filenames)), size=random_count, replace=False)
        else:
            bg_indices = np.arange(len(bg_filenames))

        for bg_ind in bg_indices:
            bg_filename = bg_filenames[bg_ind]
            file_prefix = bg_filename.split(os.sep)[-1].replace('.png', '')
            bg = cv2.imread(bg_filename)

            if min(bg.shape[:2]) < 512:
                continue

            im1, mask1 = add_line_to_image(bg, crop_height, crop_width)

            if im1 is None:
                continue
            if mask1.sum() < min(im1.shape[:2]) / 2:
                continue

            save_prefix = '%s_%d_%010d' % (file_prefix, aug_time, bg_ind)
            cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1)  # 不能有中文
            cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)

            lines.append('%s\n' % save_prefix)

            if True: #np.random.rand() < 0.01:
                cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                            np.concatenate([im1, 255 * np.stack([mask1, mask1, mask1], axis=2)],
                                           axis=1))  # 不能有中文
    if len(lines) > 0:
        with open('%s/%s.txt' % (save_root, subset), 'w') as fp:
            fp.writelines(lines)


def box_aug_v1(subset='train', aug_times=1, save_img=False, save_root=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    save_dir = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    aug_times = aug_times if subset == 'train' else 1
    gt_postfix = '_gt_5.xml'
    valid_labels_set = [1, 2, 3, 4]

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    save_txt_path = '%s/labels/' % save_dir
    for p in [save_img_path, save_txt_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    list_lines = []
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["1", "2", "3", "4"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

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
        gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix)

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                      valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        for j, (box, label) in enumerate(zip(gt_boxes, gt_labels)):  # per item
            if label <= 3:  # 1, 2, 3
                xmin0, ymin0, xmax0, ymax0 = box

                for aug_time in range(aug_times):
                    xmin, ymin, xmax, ymax = xmin0, ymin0, xmax0, ymax0
                    box_w, box_h = int(xmax - xmin), int(ymax - ymin)

                    sub_w = np.random.randint(low=max(box_w * 1.5, 600), high=max(box_w * 2, 1024))
                    sub_h = np.random.randint(low=max(box_h * 1.5, 600), high=max(box_h * 2, 1024))
                    range_x = int(sub_w - box_w)
                    range_y = int(sub_h - box_h)
                    left = np.random.randint(low=0, high=range_x)
                    up = np.random.randint(low=0, high=range_y)

                    left = xmin - left
                    up = ymin - up

                    left = max(1, left)
                    up = max(1, up)
                    if left + sub_w > orig_width:
                        sub_w = orig_width - left
                    if up + sub_h > orig_height:
                        sub_h = orig_height - up

                    xoffset = int(left)
                    yoffset = int(up)
                    sub_w = int(sub_w)
                    sub_h = int(sub_h)
                    xmin1, ymin1 = xoffset, yoffset
                    xmax1, ymax1 = xoffset + sub_w, yoffset + sub_h

                    # here, the sub_image box is [xoffset, yoffset, xoffset + sub_w, yoffset + sub_h]
                    # find all gt_boxes in this sub-rectangle
                    # 查找gtboxes里面，与当前框有交集的框
                    # idx1 = np.where(gt_labels >= 2)[0]
                    # boxes = gt_boxes[idx1, :]  # 得到框
                    # labels = gt_labels[idx1]
                    boxes = np.copy(gt_boxes)
                    labels = np.copy(gt_labels)

                    ious = box_iou_np(np.array([xmin1, ymin1, xmax1, ymax1], dtype=np.float32).reshape(-1, 4), boxes)
                    idx2 = np.where(ious > 1e-8)[1]
                    sub_gt_boxes = []
                    invalid_gt_boxes = []
                    if len(idx2) > 0:
                        valid_boxes = boxes[idx2, :]
                        valid_labels = labels[idx2]
                        valid_boxes[:, [0, 2]] -= xmin1
                        valid_boxes[:, [1, 3]] -= ymin1
                        for box1, label1 in zip(valid_boxes.astype(np.int32), valid_labels):
                            xmin, ymin, xmax, ymax = box1
                            xmin1 = max(1, xmin)
                            ymin1 = max(1, ymin)
                            xmax1 = min(xmax, sub_w - 1)
                            ymax1 = min(ymax, sub_h - 1)

                            # here, check the new gt_box[xmin1, ymin1, xmax1, ymax1]
                            # if the area of new gt_box is less than 0.6 of the original box, then remove this box and
                            # record its position, to put it to zero in the image
                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                            area = (xmax - xmin) * (ymax - ymin)
                            if area1 >= 0.6 * area:
                                sub_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                            else:
                                invalid_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                    else:
                        print('no gt boxes in this rectangle')
                        continue

                    cutout = []
                    for bi in range(3):
                        band = ds.GetRasterBand(bi + 1)
                        band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_w, win_ysize=sub_h)
                        cutout.append(band_data)
                    cutout = np.stack(cutout, -1)  # RGB
                    # cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                    # im = cv2.resize(cutout, (256, 256))  # BGR
                    # cv2.imwrite(save_filename, im)  # 不能有中文

                    if len(invalid_gt_boxes) > 0:
                        for box2 in np.array(invalid_gt_boxes).astype(np.int32):
                            xmin, ymin, xmax, ymax, label = box2
                            cutout[ymin:ymax, xmin:xmax, :] = 0

                            # check the sub_gt_boxes, remove those boxes where in invalid_gt_boxes
                            if len(sub_gt_boxes) > 0:
                                ious = box_iou_np(box2[:4].reshape(1, 4),
                                                  np.array(sub_gt_boxes, dtype=np.float32).reshape(-1, 5)[:, :4])
                                idx2 = np.where(ious > 0)[1]
                                if len(idx2) > 0:
                                    sub_gt_boxes = [box3 for ii, box3 in enumerate(sub_gt_boxes) if
                                                    ii not in idx2.tolist()]

                    # draw gt boxes
                    if len(sub_gt_boxes) > 0:
                        save_prefix = '%d_%d_%d' % (ti, j, aug_time)

                        # save image
                        # for coco format
                        single_image = {}
                        single_image['file_name'] = save_prefix + '.jpg'
                        single_image['id'] = image_id
                        single_image['width'] = sub_w
                        single_image['height'] = sub_h
                        data_dict['images'].append(single_image)

                        # for yolo format
                        cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                        list_lines.append('./images/%s.jpg\n' % save_prefix)

                        valid_lines = []
                        for box2 in sub_gt_boxes:
                            xmin, ymin, xmax, ymax, label = box2

                            if save_img:
                                cv2.rectangle(cutout, (xmin, ymin), (xmax, ymax), color=colors[label],
                                              thickness=3)

                            xc1 = int((xmin + xmax) / 2)
                            yc1 = int((ymin + ymax) / 2)
                            w1 = xmax - xmin
                            h1 = ymax - ymin

                            valid_lines.append(
                                "%d %f %f %f %f\n" % (label - 1, xc1 / sub_w, yc1 / sub_h, w1 / sub_w, h1 / sub_h))

                            # for coco format
                            single_obj = {'area': int(w1 * h1),
                                          'category_id': int(label),
                                          'segmentation': []}
                            single_obj['segmentation'].append(
                                [int(xmin), int(ymin), int(xmax), int(ymin),
                                 int(xmax), int(ymax), int(xmin), int(ymax)]
                            )
                            single_obj['iscrowd'] = 0

                            single_obj['bbox'] = int(xmin), int(ymin), int(w1), int(h1)
                            single_obj['image_id'] = image_id
                            single_obj['id'] = inst_count
                            data_dict['annotations'].append(single_obj)
                            inst_count = inst_count + 1

                        image_id = image_id + 1

                        with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                            fp.writelines(valid_lines)

                        if save_img:
                            cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                    # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                    # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                    # ims.append(im)

    if len(list_lines) > 0:
        with open(save_dir + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_dir + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


# random points according to the ground truth polygons
def aug_mc_seg_v1(subset='train', aug_times=1, save_img=False, save_root=None,
                  gt_postfixes=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    save_dir = save_root
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    aug_times = aug_times if subset == 'train' else 1

    images_root = save_dir + "/images/%s/" % subset
    labels_root = save_dir + "/annotations/%s/" % subset
    images_shown_root = save_dir + "/images_shown/%s/" % subset
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

    random_gt_ratios = [0.2, 0.8, 0.1, 0.1]
    palette = np.random.randint(0, 255, size=(len(gt_postfixes), 3))
    opacity = 0.5

    lines = []
    size0 = 10000
    size1 = -1

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

            mask_savefilename = save_dir + "/" + file_prefix + ".png"
            # cv2.imwrite(mask_savefilename, mask)
            if not os.path.exists(mask_savefilename):
                cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

        for aug_time in range(aug_times):
            for pi1, (gt_polys, gt_labels) in enumerate(zip(all_gt_polys, all_gt_labels)):
                for pi2, (poly, label) in enumerate(zip(gt_polys, gt_labels)):  # poly为nx2的点, numpy.array
                    cx, cy = np.mean(poly, axis=0).astype(np.int32)
                    pw = np.max(poly[:, 0]) - np.min(poly[:, 0])
                    ph = np.max(poly[:, 1]) - np.min(poly[:, 1])
                    ex = np.random.randint(low=int(0.5*pw), high=int(2.0*pw))
                    ey = np.random.randint(low=int(0.5*ph), high=int(2.0*ph))
                    xoffset = cx - pw//2 - ex
                    sub_width = pw + 2*ex
                    yoffset = cy - ph//2 - ey
                    sub_height = ph + 2*ey

                    if sub_width < 512:
                        xoffset = cx - np.random.randint(low=256, high=512)
                        sub_width = np.random.randint(low=512, high=1024)
                    if sub_height < 512:
                        yoffset = cy - np.random.randint(low=256, high=512)
                        sub_height = np.random.randint(low=512, high=1024)

                    xoffset = max(1, xoffset)
                    yoffset = max(1, yoffset)
                    if xoffset + sub_width > orig_width - 1:
                        sub_width = orig_width - 1 - xoffset
                    if yoffset + sub_height > orig_height - 1:
                        sub_height = orig_height - 1 - yoffset
                    xoffset, yoffset, sub_width, sub_height = [int(val) for val in
                                                               [xoffset, yoffset, sub_width, sub_height]]
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
                    seg = mask[(yoffset):(yoffset + sub_height), (xoffset):(xoffset + sub_width)]
                    seg_count = len(np.where(seg > 0)[0])
                    if seg_count < 10:
                        continue

                    assert img.shape[:2] == seg.shape[:2]

                    H, W = img.shape[:2]
                    num_x_splits = max(1, int(np.round(W/1024)))
                    num_y_splits = max(1, int(np.round(H/1024)))
                    x_splits = np.array_split(np.arange(W, dtype=np.int32), num_x_splits)
                    y_splits = np.array_split(np.arange(H, dtype=np.int32), num_y_splits)

                    for sx in range(num_x_splits):
                        x1, x2 = x_splits[sx][0], x_splits[sx][-1]+1
                        for sy in range(num_y_splits):
                            y1, y2 = y_splits[sy][0], y_splits[sy][-1]+1

                            img1 = img[y1:y2, x1:x2, ::-1]
                            seg1 = seg[y1:y2, x1:x2]

                            seg1_count = len(np.where(seg1 > 0)[0])
                            if seg1_count < 10:
                                continue

                            if len(np.where(img1[:,:,0]==0)[0]) > 0.4 * np.prod(img1.shape[:2]) \
                                    or len(np.where(img1[:,:,0]==255)[0]) > 0.4 * np.prod(img1.shape[:2]):
                                continue

                            size0 = min(size0, min(img1.shape[:2]))
                            size1 = max(size1, max(img1.shape[:2]))

                            save_prefix = '%03d_%d_%d_%d_%d_%d' % (ti, aug_time, pi1, pi2, sx, sy)
                            cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), img1)  # 不能有中文
                            cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), seg1)

                            lines.append('%s\n' % save_prefix)

                            if np.random.rand() < 0.01:
                                # cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                                #             np.concatenate([im1, 255 * np.stack([mask1, mask1, mask1], axis=2)],
                                #                            axis=1))  # 不能有中文
                                color_seg = np.zeros((seg1.shape[0], seg1.shape[1], 3), dtype=np.uint8)
                                for label, color in enumerate(palette):
                                    color_seg[seg1 == (label + 1), :] = color
                                # convert to BGR
                                color_seg = color_seg[..., ::-1]

                                img1 = img1 * (1 - opacity) + color_seg * opacity
                                img1 = img1.astype(np.uint8)
                                cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix), img1)
                    del img, seg
        del mask


    if len(lines) > 0:
        with open(save_root + '/%s_%d_%d.txt' % (subset, size0, size1), 'w') as fp:
            fp.writelines(lines)


def get_args_parser():
    parser = argparse.ArgumentParser('gd augmentation', add_help=False)
    parser.add_argument('--cached_data_path', default='', type=str)
    parser.add_argument('--subset', default='train', type=str)
    parser.add_argument('--aug_type', default='', type=str)
    parser.add_argument('--aug_times', default=1, type=int)
    parser.add_argument('--random_count', default=1, type=int)
    parser.add_argument('--save_img', default=False, action='store_true')
    parser.add_argument('--do_rotate', default=False, action='store_true')
    parser.add_argument('--crop_height', default=0, type=int)
    parser.add_argument('--crop_width', default=0, type=int)
    parser.add_argument('--update_cache', default=False, action='store_true')

    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser('gd augmentation', parents=[get_args_parser()])
    args = parser.parse_args()

    hostname = socket.gethostname()
    subset = args.subset
    cached_data_path = args.cached_data_path
    aug_type = args.aug_type
    aug_times = args.aug_times
    do_rotate = args.do_rotate
    save_img = args.save_img
    random_count = args.random_count
    crop_height = args.crop_height
    crop_width = args.crop_width
    update_cache = args.update_cache

    # for semantic segmentation
    if aug_type == 'mc_seg_v1':
        # '_gt_building7.xml',
        # '_gt_landslide10.xml',
        # '_gt_water6.xml'
        # '_gt_tree8.xml',
        # '_gt_flood12.xml'
        gt_postfixes, gt_name = ['_gt_water6.xml'], 'water6'
        gt_postfixes, gt_name = ['_gt_building7.xml'], 'building7'
        gt_postfixes, gt_name = ['_gt_landslide10.xml'], 'landslide10'
        gt_postfixes, gt_name = ['_gt_road9.xml'], 'road9'
        gt_postfixes = [
            '_gt_building7.xml',
            '_gt_water6.xml',
            '_gt_road9.xml',
            '_gt_landslide10.xml'
        ]
        gt_name = '4classes'

        if hostname == 'master':
            save_root = '/media/ubuntu/Data/gd_mc_seg_Aug%d/%s_%s/' % (aug_times, aug_type, gt_name)
        else:
            save_root = 'E:/gd_mc_seg_Aug%d/%s_%s/' % (aug_times, aug_type, gt_name)

        aug_mc_seg_v1(subset=subset, aug_times=aug_times, save_img=save_img, save_root=save_root,
                      gt_postfixes=gt_postfixes)
        sys.exit(-1)


    # for detection aug
    if hostname == 'master':
        save_root = '/media/ubuntu/Data/gd_newAug%d_Rot%d_4classes' % (aug_times, do_rotate)
    else:
        save_root = 'E:/gd_newAug%d_Rot%d_4classes' % (aug_times, do_rotate)

    if aug_type == 'check_fg_images_v1':
        save_root = '%s/%s' % (save_root, aug_type)
        check_fg_images(subset=subset, save_root=save_root)
        sys.exit(-1)

    if aug_type == 'check_dataset':
        check_dataset(subset=subset)
        sys.exit(-1)

    """
    cached_data_path/train_fg_images.npy
    cached_data_path/train_fg_boxes.npy
    cached_data_path/val_fg_images.npy
    cached_data_path/val_fg_boxes.npy
    cached_data_path/train_bg_images/
    cached_data_path/val_bg_images/
    """
    print('extract fg images ...')
    # save_dir = 'E:/fg_images_shown/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    fg_images_filename, fg_boxes_filename = extract_fg_images(subset, cached_data_path,
                                                              do_rotate=do_rotate,
                                                              update_cache=update_cache,
                                                              debug=False)

    # sys.exit(-1)

    if False:
        fg_images_list = np.load(fg_images_filename, allow_pickle=True)  # list of RGB images [HxWx3]
        fg_boxes_list = np.load(fg_boxes_filename, allow_pickle=True)  # list of boxes [nx5]
        for fi, (img, boxes) in enumerate(zip(fg_images_list, fg_boxes_list)):
            for box in boxes.astype(np.int32):
                x1, y1, x2, y2, label = box
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
                cv2.putText(img, str(label), (x1, y1), fontFace=1, fontScale=1, color=(0, 255, 255), thickness=1)
            cv2.imwrite('%s/%10d.jpg'%(save_dir, fi), img)
            pass

    print('extract bg images ...')
    bg_images_dir = extract_bg_images(subset, cached_data_path, random_count, update_cache=update_cache)

    print('doing augmentation ...')
    if aug_type == 'box_aug_v1':
        # TODO zzs, implement the rotate augmentation
        # in this augmentation, just use the crop with random shift
        # save_root/box_aug_v1/train/images/*.jpg
        # save_root/box_aug_v1/train/labels/*.txt
        # save_root/box_aug_v1/train/train.txt
        # save_root/box_aug_v1/val/images/*.jpg
        # save_root/box_aug_v1/val/labels/*.txt
        # save_root/box_aug_v1/val/val.txt
        save_root = '%s/%s' % (save_root, aug_type)
        box_aug_v1(subset=subset, aug_times=aug_times, save_img=save_img, save_root=save_root)

    elif aug_type == 'box_aug_v2':
        # TODO zzs, to generate more fg image patches, add rotation compose

        # save_root/box_aug_v2/train/images/*.jpg
        # save_root/box_aug_v2/train/labels/*.txt
        # save_root/box_aug_v2/train/train.txt
        # save_root/box_aug_v2/val/images/*.jpg
        # save_root/box_aug_v2/val/labels/*.txt
        # save_root/box_aug_v2/val/val.txt
        save_root = '%s/%s' % (save_root, aug_type)
        compose_fg_bg_images(subset=subset, aug_times=aug_times, save_root=save_root,
                             fg_images_filename=fg_images_filename,
                             fg_boxes_filename=fg_boxes_filename,
                             bg_images_dir=bg_images_dir)

    elif aug_type == 'refine_line_v1':
        # TODO zzs need to generate the parallel lines like the wire

        # save_root/refine_line_v1_512_512/train/images/*.jpg
        # save_root/refine_line_v1_512_512/train/annotations/*.png
        # save_root/refine_line_v1_512_512/train/train.txt
        # save_root/refine_line_v1_512_512/val/images/*.jpg
        # save_root/refine_line_v1_512_512/val/annotations/*.png
        # save_root/refine_line_v1_512_512/val/val.txt
        save_root = '%s/%s_%d_%d' % (save_root, aug_type, crop_height, crop_width)
        refine_line_aug(subset=subset, aug_times=aug_times, save_root=save_root,
                        crop_height=crop_height, crop_width=crop_width,
                        fg_images_filename=fg_images_filename,
                        fg_boxes_filename=fg_boxes_filename,
                        bg_images_dir=bg_images_dir, random_count=random_count)
    else:
        print('wrong aug type')
        sys.exit(-1)
