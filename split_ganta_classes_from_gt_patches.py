import sys, os, glob


sys.path.insert(0, 'D:/rs/gd/yoloV5/')

import cv2
import numpy as np
import torch
from osgeo import gdal, osr
from natsort import natsorted
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, save_predictions_to_envi_xml
from utils.general import xyxy2xywh, xywh2xyxy, box_iou

"""
最开始的标注只有两类，1：杆塔，2：绝缘子。但是训练模型发现对于杆塔效果不好
有的大的杆塔，有的二根杆子的杆塔，有的是民用的杆塔
于是需要对1进行细分。
首先用extract_gt_patches.py提取所有1的图片。然后人工分成3类。
把对应的图片移动到对应的文件夹。
然后用本文件的代码，生成新的xml文件。
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
    for label in [1, 2]:
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


def main(subset='train'):
    source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
    gt_dir = 'E:/gd_gt_combined'  # sys.argv[2]
    new_gt_dir = 'E:/gd_gt_combined_4classes'   # 3 杆塔+1绝缘子
    save_root = 'E:/patches_gt/%s' % (subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(new_gt_dir):
        os.makedirs(new_gt_dir)

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
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_new.xml')

        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform)
        gt_boxes = torch.from_numpy(gt_boxes)
        gt_labels = torch.from_numpy(gt_labels)

        boxes = gt_boxes
        labels = gt_labels

        assert len(boxes) == len(labels), 'check boxes'
        assert len(gt_boxes) == len(gt_labels), 'check gt_boxes'

        save_dir = os.path.join(save_root, '%d'%ti)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_sub_dirs = ['small', 'median', 'large']
        for save_sub_dir in save_sub_dirs:
            save_sub_dir = os.path.join(save_dir, save_sub_dir)
            if not os.path.exists(save_sub_dir):
                os.makedirs(save_sub_dir)

        # 1:small, 2:median, 3:large, 4:jueyuanzi
        gt_labels[np.where(gt_labels==2)[0]]=4
        for subi, save_sub_dir in enumerate(save_sub_dirs):
            save_sub_dir = os.path.join(save_dir, save_sub_dir)
            img_files = glob.glob(save_sub_dir + '/*.jpg')
            inds = [img_file.split(os.sep)[-1].replace('.jpg','')
                    for img_file in img_files]
            inds = np.array([int(float(ind)) for ind in inds])
            gt_labels[inds] = subi + 1

        # 写入新的xml文件
        preds = np.concatenate([gt_boxes.reshape([-1, 4]), gt_labels.reshape([-1, 1])], axis=1)
        save_xml_filename = os.path.join(new_gt_dir, file_prefix+'_gt_new.xml')
        save_predictions_to_envi_xml(preds, save_xml_filename,
                                     gdal_proj_info=projection_esri,
                                     gdal_trans_info=geotransform,
                                     names={0: '1', 1:'2', 2:'3', 3:'4'},
                                     colors={0: "255,0,0",
                                             1: "255,255,0",
                                             2: "0,255,255",
                                             3: "0,255,0"})

if __name__ == '__main__':
    main(subset='train')
    # main(subset='val')
