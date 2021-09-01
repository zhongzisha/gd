import shutil
import sys, os, glob
import argparse
import cv2
import numpy as np
import torch
from osgeo import gdal, osr, ogr
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

LABELS_TO_NAMES = {
    3: 'tower',
    4: 'insulator',
    5: 'line_box',
    6: 'water',
    7: 'building',
    8: 'tree',
    9: 'road',
    10: 'landslide',
    14: 'line_region'
}

NAMES_TO_LABELS = {v: k for k, v in LABELS_TO_NAMES.items()}


def load_polys_from_shpfile(tif_filename, shp_filename, valid_labels):

    ds = gdal.Open(tif_filename, gdal.GA_ReadOnly)

    gdal_trans_info = ds.GetGeoTransform()
    projection = ds.GetProjection()

    xOrigin = gdal_trans_info[0]
    yOrigin = gdal_trans_info[3]
    pixelWidth = gdal_trans_info[1]
    pixelHeight = gdal_trans_info[5]

    shp_driver = ogr.GetDriverByName("ESRI Shapefile")

    shp_ds = shp_driver.Open(shp_filename, update=0)
    layerCount = shp_ds.GetLayerCount()

    layer = shp_ds.GetLayer(0)
    layerDef = layer.GetLayerDefn()
    for i in range(layerDef.GetFieldCount()):
        fieldDef = layerDef.GetFieldDefn(i)
        fieldName = fieldDef.GetName()
        fieldTypeCode = fieldDef.GetType()
        fieldType = fieldDef.GetFieldTypeName(fieldTypeCode)
        fieldWidth = fieldDef.GetWidth()
        fieldPrecision = fieldDef.GetPrecision()

        print(i, fieldName, fieldType, fieldWidth, fieldPrecision)

    polys = []
    for i, feat in enumerate(layer):
        geom = feat.GetGeometryRef()
        label = NAMES_TO_LABELS[feat.GetField(0)]
        if label in valid_labels:
            wkt_str = geom.ExportToWkt()
            start = wkt_str.find("((") + 2
            end = wkt_str.find("))")
            points = []
            for point in wkt_str[start:end].split(','):
                x, y = point.split(' ')
                x = (float(x) - xOrigin) / pixelWidth + 0.5
                y = (float(y) - yOrigin) / pixelHeight + 0.5
                points += [x, y]
            polys.append(np.array(points).reshape(-1, 2).astype(np.int32))
    shp_ds = None
    ds = None

    return polys
        

def prepare_landslide_segmentation_dataset(save_root):
    hostname = socket.gethostname()
    source = 'E:/all_tif_files.txt'

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    tiffiles = [r'G:\gddata\all\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif']

    for tiffile in tiffiles:
        process_one_tif(save_root, tiffile)


# random points according to the ground truth polygons
def process_one_tif(save_root=None, tiffile=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        gt_dir = r'E:\gddata_processed'  # sys.argv[2]

    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
    print(file_prefix)

    # valid_labels_set = [1, 2, 3, 4]
    valid_labels_set = [10]
    palette = np.random.randint(0, 255, size=(len(valid_labels_set), 3))  # building, water, road, landslide
    palette = np.array([[255, 255, 255], [250, 0, 0]])
    opacity = 0.2

    save_dir = '%s/%s/' % (save_root, file_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    save_lbl_path = '%s/labels/' % save_dir
    for p in [save_img_path, save_lbl_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    for label_ind, label in enumerate(valid_labels_set):
        label_name = LABELS_TO_NAMES[label]
        shp_filename = os.path.join(gt_dir, file_prefix + '_gt_%s.shp' % (label_name))

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

        gt_polys = load_polys_from_shpfile(tiffile, shp_filename, valid_labels=[label])
        # print(gt_polys)
        if len(gt_polys) == 0:
            continue

        # 首先根据标注生成mask图像，存在内存问题！！！
        print('generate mask ...')
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
        # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)
        for poly in gt_polys:  # poly为nx2的点, numpy.array
            print(poly)
            print(len(poly))
            cv2.drawContours(mask, [poly], -1, color=(label, label, label), thickness=-1)
        time.sleep(5)

        offsets = compute_offsets(height=orig_height, width=orig_width, subsize=1024, gap=0)

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

            # sample points from mask
            seg = mask[(yoffset):(yoffset + sub_height), (xoffset):(xoffset + sub_width)]
            # seg1_count = len(np.where(seg > 0)[0])
            # if seg1_count < 10:
            #     continue

            # print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
            img = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
            for b in range(3):
                band = ds.GetRasterBand(b + 1)
                img[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)

            img_sum = np.sum(img, axis=2)
            indices_y, indices_x = np.where(img_sum > 0)
            if len(indices_x) == 0:
                continue

            assert img.shape[:2] == seg.shape[:2]

            img1 = img.copy()
            if len(np.where(img1[:, :, 0] == 0)[0]) > 0.4 * np.prod(img1.shape[:2]) \
                    or len(np.where(img1[:, :, 0] == 255)[0]) > 0.4 * np.prod(img1.shape[:2]):
                continue

            minsize = min(img1.shape[:2])
            maxsize = max(img1.shape[:2])
            if maxsize > 1.5 * minsize:
                continue

            save_prefix = '%s_%d_%d' % (file_prefix, int(xoffset), int(yoffset))
            # cv2.imwrite('%s/%s.jpg' % (save_img_path, save_prefix), img)  # 不能有中文
            # cv2.imwrite('%s/%s.png' % (save_lbl_path, save_prefix), seg)
            Image.fromarray(img).save('%s/%s.jpg' % (save_img_path, save_prefix))
            Image.fromarray(seg).save('%s/%s.png' % (save_lbl_path, save_prefix))

            if True:
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                color_seg[seg == label, :] = palette[label_ind]

                img1 = img1 * (1 - opacity) + color_seg * opacity
                img1 = img1.astype(np.uint8)
                # cv2.imwrite('%s/%s.jpg' % (save_img_shown_path, save_prefix), img1)
                Image.fromarray(img1).save('%s/%s.jpg' % (save_img_shown_path, save_prefix))


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
        save_root = '/media/ubuntu/Data/gd_newAug%d_Rot%d_4classes_landslide_segmentation' % (aug_times, do_rotate)
    else:
        save_root = r'E:\gd_newAug%d_Rot%d_4classes_landslide_segmentation_test' % (aug_times, do_rotate)

    if aug_type == 'landslide_segmentation':
        save_root = '%s/%s' % (save_root, aug_type)
        prepare_landslide_segmentation_dataset(save_root=save_root)
        sys.exit(-1)
