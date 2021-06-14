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
python extract_image_patches2.py ^
--source E:\train1_list.txt ^
--subset train1 ^
--pred_dir E:\mmdetection\work_dirs\faster_rcnn_r50_fpn_dc5_1x_coco_lr0.001_newAug4_v2_new\outputs_train1_1024_32_epoch_6 ^
--save_root E:\ganta_patch_classification ^
--gt_dir G:\gddata\aerial
"""


def get_args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--source", type=str, default='E:/train_list.txt')
    parser.add_argument("--subset", type=str, default='train')
    parser.add_argument("--pred_dir", type=str, default='')
    parser.add_argument("--save_root", type=str, default='')
    parser.add_argument("--gt_dir", type=str, default='')

    return parser.parse_args()


def main(args):
    source = args.source
    subset = args.subset
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    save_root = args.save_root

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

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename,
                                                    gdal_trans_info=geotransform,
                                                    valid_labels=[1,2,3,4])
        gt_boxes = torch.from_numpy(gt_boxes)
        gt_labels = torch.from_numpy(gt_labels)
        gt_scores = torch.ones_like(gt_labels, dtype=scores.dtype)

        print('number of boxes: ', len(boxes))
        print('number of gt boxes: ', len(gt_boxes))

        boxes = torch.cat([boxes, gt_boxes], dim=0)
        scores = torch.cat([scores, gt_scores], dim=0)
        labels = torch.cat([labels, gt_labels], dim=0)

        assert len(boxes) == len(labels), 'check boxes'
        assert len(gt_boxes) == len(gt_labels), 'check gt_boxes'

        if True:
            # 提取image patches
            # Reshape and pad cutouts
            b = xyxy2xywh(boxes[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 32  # pad
            # d[:, :4] = xywh2xyxy(b).long()

            ious = box_iou(boxes[:, :4], gt_boxes[:, :4])

            # import pdb
            # pdb.set_trace()

            ims = []
            for j, (a, score, label) in enumerate(zip(b, scores, labels)):  # per item
                if label == 3:

                    # 在这里区分pos和neg，设定iou阈值，与gt的iou超过阈值认为正样本，反之负样本
                    if ious[j].max() > 0.1:
                        save_filename = '%s/pos/%03d_%06d_%.3f.jpg' % (save_root, ti, j, score)
                        lines.append('%s 1\n' % save_filename)
                    else:
                        save_filename = '%s/neg/%03d_%06d_%.3f.jpg' % (save_root, ti, j, score)
                        lines.append('%s 0\n' % save_filename)

                    cutout = []
                    xc, yc = int(a[0]), int(a[1])
                    width, height = int(a[2]), int(a[3])
                    xoffset = max(0, xc - width // 2)
                    yoffset = max(0, yc - height // 2)
                    if xoffset + width > orig_width:
                        width = orig_width - xoffset
                    if yoffset + height > orig_height:
                        height = orig_height - yoffset
                    for bi in range(3):
                        band = ds.GetRasterBand(bi + 1)
                        band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=width, win_ysize=height)
                        cutout.append(band_data)
                    cutout = np.stack(cutout, -1)  # RGB
                    # cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                    im = cv2.resize(cutout, (256, 256))  # BGR
                    cv2.imwrite(save_filename, im)  # 不能有中文

                    # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                    # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                    # ims.append(im)

    if len(lines) > 0:
        with open(save_root + '/%s_list.txt' % subset, 'w') as fp:
            fp.writelines(lines)


if __name__ == '__main__':
    args = get_args()
    main(args)
