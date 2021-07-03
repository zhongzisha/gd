import sys, os, glob
import argparse

# sys.path.insert(0, 'F:/gd/yoloV5/')

import cv2
import numpy as np
import torch
from osgeo import gdal, osr
from natsort import natsorted
from myutils import py_cpu_nms, xyxy2xywh, xywh2xyxy, box_iou, load_gt_for_detection

"""
从大图检测的结果，对于每个大图像a.tif都会保存a.tif, a.xml, a_all_preds.pt
其中，a.xml是经过了nms的all_preds, a_all_preds.pt是没有经过nms的.
a.xml是ENVI格式的ROI文件。
"""

"""
python det_results_analysis.py ^
--source E:\train1_list.txt ^
--subset train1 ^
--pred_dir E:\mmdetection\work_dirs\faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_newAug4_v2_new\outputs_train1_1024_32_epoch_6 ^
--save_root E:\ganta_patch_classification\ ^
--gt_dir G:\gddata\aerial ^
--save_postfix ''

python det_results_analysis.py ^
--source E:\val1_list.txt ^
--subset val1 ^
--pred_dir E:\mmdetection\work_dirs\faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_newAug4_v2_new\outputs_val1_1024_32_epoch_6 ^
--save_root E:\ganta_patch_classification\ ^
--gt_dir G:\gddata\aerial ^
--save_postfix ''

python det_results_analysis.py ^
--source E:\val1_list.txt ^
--subset val1 ^
--pred_dir E:\mmdetection\work_dirs\faster_rcnn_r50_fpn_dc5_2x_coco_lr0.001_newAug4_v2_new_v2_onlyAug3_rotate\outputs_val1_1024_32_epoch_24 ^
--save_root E:\mmdetection\work_dirs\faster_rcnn_r50_fpn_dc5_2x_coco_lr0.001_newAug4_v2_new_v2_onlyAug3_rotate\outputs_val1_1024_32_epoch_24 ^
--gt_dir G:\gddata\aerial ^
--save_postfix ''

python det_results_analysis.py ^
--source E:\train1_list.txt ^
--subset train1 ^
--pred_dir E:\mmdetection\work_dirs\faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_newAug4_v2_new\outputs_train1_1024_32_epoch_6_withCls ^
--save_root E:\ganta_patch_classification\ ^
--gt_dir G:\gddata\aerial ^
--save_postfix '_withCls'

python det_results_analysis.py ^
--source E:\val1_list.txt ^
--subset val1 ^
--pred_dir E:\mmdetection\work_dirs\faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_newAug4_v2_new\outputs_val1_1024_32_epoch_6_withCls ^
--save_root E:\ganta_patch_classification\ ^
--gt_dir G:\gddata\aerial ^
--save_postfix '_withCls'
"""


def get_args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--source", type=str, default='E:/train_list.txt')
    parser.add_argument("--subset", type=str, default='train')
    parser.add_argument("--pred_dir", type=str, default='')
    parser.add_argument("--save_root", type=str, default='')
    parser.add_argument("--gt_dir", type=str, default='')
    parser.add_argument("--save_postfix", type=str, default='')

    return parser.parse_args()


def main(args):
    source = args.source
    subset = args.subset
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    save_root = os.path.join(args.save_root, subset)
    save_postfix = args.save_postfix.replace('\'', '')

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for n in ['pos', 'neg']:
        save_dir = os.path.join(save_root, n)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    lines = ["name,num_gt,num_pred,num_tp\n"]

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

        pred_filename = os.path.join(pred_dir, file_prefix + '_all_preds.pt')
        if not os.path.exists(pred_filename):
            continue

        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        all_preds = torch.load(pred_filename).cpu()
        tmp_preds = []
        all_preds_cpu = all_preds.numpy()
        for label in [0, 1, 2, 3]:
            idx = np.where(all_preds_cpu[:, 5] == label)[0]
            if len(idx) > 0:
                # if label == 0:
                #     pw = all_preds_cpu[idx, 2] - all_preds_cpu[idx, 0]
                #     ph = all_preds_cpu[idx, 3] - all_preds_cpu[idx, 1]
                #     inds = np.where((pw >= 100) | (ph >= 100))[0]
                #     valid_inds = idx[inds]
                # elif label == 1:
                #     pw = all_preds_cpu[idx, 2] - all_preds_cpu[idx, 0]
                #     ph = all_preds_cpu[idx, 3] - all_preds_cpu[idx, 1]
                #     inds = np.where((pw < 100) & (ph < 100))[0]
                #     valid_inds = idx[inds]
                valid_inds = idx
                dets = all_preds_cpu[valid_inds, :5]
                keep = py_cpu_nms(dets, thresh=0.5)
                tmp_preds.append(all_preds[valid_inds[keep]])
        all_preds = torch.cat(tmp_preds)
        boxes, scores, labels = all_preds[:, :4], all_preds[:, 4], all_preds[:, 5] + 1

        if len(boxes) == 0:
            continue

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename,
                                                    gdal_trans_info=geotransform,
                                                    valid_labels=[1,2,3,4])

        if len(gt_boxes) > 0:
            gt_boxes = torch.from_numpy(gt_boxes)
            gt_labels = torch.from_numpy(gt_labels)
            gt_scores = torch.ones_like(gt_labels, dtype=scores.dtype)
        else:
            continue

        gt_boxes = gt_boxes[gt_labels == 3, :]
        boxes = boxes[labels == 3, :]

        ious = box_iou(boxes[:, :4], gt_boxes[:, :4])  # NxM

        tp = 0
        for j, (box, score, label) in enumerate(zip(boxes, scores, labels)):  # per item
            if ious[j].max() > 0:  # 只统计杆塔
                tp += 1

        lines.append("%s,%d,%d,%d\n"%(file_prefix, len(gt_boxes), len(boxes), tp))

    if len(lines) > 0:
        with open(save_root + '/%s_result_analysis%s.csv' % (subset, save_postfix), 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    args = get_args()
    main(args)
