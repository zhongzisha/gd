import shutil
import sys, os, glob
import argparse
import cv2
import numpy as np
import torch
from osgeo import gdal, osr
from natsort import natsorted
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, box_iou_np, \
    box_intersection_np, load_gt_polys_from_esri_xml, compute_offsets, alpha_map, elastic_transform_v2, \
    load_gt_for_detection
import json
import socket
from PIL import Image, ImageDraw, ImageFilter
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import shapely.geometry as shgeo
import time


def prepare_super_resolution_dataset(subset='train', save_root=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/super_resolution_%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        source = 'E:/super_resolution_%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'G:/gddata/all'  # sys.argv[2]

    info_filename = os.path.join(gt_dir, '{}_infos.csv'.format(subset))
    # if os.path.exists(info_filename):
    #     return -1

    save_root = os.path.join(save_root, subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

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

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        save_dir = os.path.join(save_root, file_prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                    valid_labels=valid_labels_set)

        # mask_ds = gdal.Open(mask_savefilename, gdal.GA_ReadOnly)
        offsets = compute_offsets(height=orig_height, width=orig_width, subsize=1920, gap=0)

        for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up
            # sub_width = min(orig_width, big_subsize)
            # sub_height = min(orig_height, big_subsize)
            # if xoffset + sub_width > orig_width:
            #     sub_width = orig_width - xoffset
            # if yoffset + sub_height > orig_height:
            #     sub_height = orig_height - yoffset
            print(oi, len(offsets), xoffset, yoffset, sub_width, sub_height)

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

            img1 = img
            # img1 = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            img1_sum = np.sum(img1, axis=2)
            indices_y, indices_x = np.where(img1_sum > 0)
            if len(indices_x) == 0:
                continue

            img1 = img1[:, :, ::-1]
            if len(np.where(img1[:, :, 0] == 0)[0]) > 0.4 * np.prod(img1.shape[:2]) \
                    or len(np.where(img1[:, :, 0] == 255)[0]) > 0.4 * np.prod(img1.shape[:2]):
                continue

            save_prefix = '%03d_%d' % (ti, oi)
            # cv2.imwrite('%s/%s.jpg' % (save_dir, save_prefix), img1)  # 不能有中文
            cv2.imencode('.png', img1)[1].tofile('%s/%s.png' % (save_dir, save_prefix))


def prepare_super_resolution_dataset_check1(subset='train', save_root=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/super_resolution_%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        source = 'E:/super_resolution_%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'G:/gddata/all'  # sys.argv[2]

    save_root = os.path.join(save_root, subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        save_dir = os.path.join(save_root, file_prefix)

        filenames = glob.glob(os.path.join(save_dir, '*.png'))
        notgood_dir = os.path.join(save_dir, 'notgood')
        invalid_files = []
        for filename in filenames:
            im = Image.open(filename)
            if im.size != (1920, 1920):
                invalid_files.append(filename)
                # shutil.move(filename, os.path.join(notgood_dir, filename.split(os.sep)[-1]))

        im = None
        for filename in invalid_files:
            shutil.move(filename, os.path.join(notgood_dir, filename.split(os.sep)[-1]))
            print(ti, filename)


def prepare_super_resolution_dataset_gen_lr(subset='train', save_root=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/super_resolution_%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        source = 'E:/super_resolution_%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'G:/gddata/all'  # sys.argv[2]

    save_root = os.path.join(save_root, subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    new_save_dir = os.path.join(save_root, 'HR')
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    scales = [2, 3, 4]
    for scale in scales:
        new_save_dir = os.path.join(save_root, 'LR_X%d' % scale)
        if not os.path.exists(new_save_dir):
            os.makedirs(new_save_dir)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        save_dir = os.path.join(save_root, file_prefix)

        filenames = glob.glob(os.path.join(save_dir, '*.png'))
        # notgood_dir = os.path.join(save_dir, 'notgood')
        # invalid_files = []
        # for filename in filenames:
        #     im = Image.open(filename)
        #     if im.size != (1920, 1920):
        #         invalid_files.append(filename)
        #         # shutil.move(filename, os.path.join(notgood_dir, filename.split(os.sep)[-1]))
        #
        # im = None
        # for filename in invalid_files:
        #     shutil.move(filename, os.path.join(notgood_dir, filename.split(os.sep)[-1]))
        #     print(ti, filename)

        for filename in filenames:
            print(ti, filename)
            im = Image.open(filename)
            size = im.size
            basename = filename.split(os.sep)[-1]
            im.save(os.path.join(save_root, 'HR', basename))
            for scale in scales:
                new_save_dir = os.path.join(save_root, 'LR_X%d' % scale)
                im.resize((size[0]//scale, size[1]//scale), Image.BICUBIC).save(os.path.join(new_save_dir, basename))


def get_args_parser():
    parser = argparse.ArgumentParser('gd augmentation', add_help=False)
    parser.add_argument('--cached_data_path', default='', type=str)
    parser.add_argument('--subset', default='train', type=str)
    parser.add_argument('--aug_type', default='', type=str)
    parser.add_argument('--postfix', default='', type=str)
    parser.add_argument('--aug_times', default=1, type=int)
    parser.add_argument('--random_count', default=1, type=int)
    parser.add_argument('--save_img', default=False, action='store_true')
    parser.add_argument('--do_rotate', default=False, action='store_true')
    parser.add_argument('--crop_height', default=512, type=int)
    parser.add_argument('--crop_width', default=512, type=int)
    parser.add_argument('--update_cache', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')

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
    debug = args.debug

    # for detection aug
    if hostname == 'master':
        save_root = '/media/ubuntu/Data/gd_newAug%d_Rot%d_4classes_super_resolution' % (aug_times, do_rotate)
    else:
        save_root = r'E:\gd_newAug%d_Rot%d_4classes_super_resolution' % (aug_times, do_rotate)

    if aug_type == 'super_resolution':
        save_root = '%s/%s' % (save_root, aug_type)
        prepare_super_resolution_dataset(subset=subset, save_root=save_root)
        sys.exit(-1)

    if aug_type == 'super_resolution_check1':
        save_root = r'%s\super_resolution' % (save_root)
        prepare_super_resolution_dataset_check1(subset=subset, save_root=save_root)
        sys.exit(-1)

    if aug_type == 'super_resolution_gen_lr':
        save_root = r'%s\super_resolution' % (save_root)
        prepare_super_resolution_dataset_gen_lr(subset=subset, save_root=save_root)
        sys.exit(-1)