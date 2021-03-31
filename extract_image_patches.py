import sys, os, glob

sys.path.insert(0, 'D:/rs/gd/yoloV5/')

import cv2
import numpy as np
import torch
from osgeo import gdal, osr
from natsort import natsorted
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms
from utils.general import xyxy2xywh, xywh2xyxy, box_iou

"""
从大图检测的结果，对于每个大图像a.tif都会保存a.tif, a.xml, a_all_preds.pt
其中，a.xml是经过了nms的all_preds, a_all_preds.pt是没有经过nms的.
a.xml是ENVI格式的ROI文件。
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
    pred_dir = 'E:/detect_outputs_yolov5x_aerial2'  # sys.argv[2]
    gt_dir = 'E:/gd_gt'  # sys.argv[2]
    save_root = 'E:/patches/%s' % (subset)
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

        pred_xml_filename = os.path.join(pred_dir, file_prefix + '.xml')
        if not os.path.exists(pred_xml_filename):
            continue

        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_new.xml')

        boxes, labels = load_gt_from_esri_xml(pred_xml_filename, gdal_trans_info=geotransform,
                                              mapcoords2pixelcoords=True)
        gt_boxes, gt_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform)
        gt_boxes = torch.from_numpy(gt_boxes)
        gt_labels = torch.from_numpy(gt_labels)

        print('number of boxes: ', len(boxes))
        print('number of gt boxes: ', len(gt_boxes))

        boxes = torch.from_numpy(np.array(boxes))
        labels = torch.from_numpy(np.array(labels))

        boxes = torch.cat([boxes, gt_boxes], dim=0)
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
            for j, (a, label) in enumerate(zip(b, labels)):  # per item
                if label == 1:

                    # 在这里区分pos和neg，设定iou阈值，与gt的iou超过阈值认为正样本，反之负样本
                    if ious[j].max() > 0.1:
                        save_filename = '%s/pos/%010d.jpg' % (save_root, j)
                        lines.append('%s 1\n' % save_filename)
                    else:
                        save_filename = '%s/neg/%010d.jpg' % (save_root, j)
                        lines.append('%s 0\n' % save_filename)

                    cutout = []
                    xc, yc = int(a[0]), int(a[1])
                    width, height = int(a[2]), int(a[3])
                    xoffset = max(0, xc - width // 2)
                    yoffset = max(0, yc - height // 2)
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
    main(subset='val')
