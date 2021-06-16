
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
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, \
    compute_offsets, save_predictions_to_envi_xml, LoadImages, \
    xyxy2xywh, xywh2xyxy, box_iou

"""
从*_gt_5.xml标注文件中提取第5类标签，提取image patches进行块分类
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
    for label in [5]: #[1, 2, 3, 4, 5]:
        idx = np.where(all_boxes[:, 4] == label)[0]
        if len(idx) > 0:
            boxes_thisclass = all_boxes[idx, :4]
            labels_thisclass = all_boxes[idx, 4]
            if label < 5:
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


def main(subset='train'):
    source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
    gt_dir = 'G:/gddata/aerial'    # *_gt_5.xml保存在这个目录下了
    save_root = 'E:/line_patches512_for_testing/%s' % (subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root + "/test/images")
        os.makedirs(save_root + "/test/annotations")

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

    prefixes = []

    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[240:270, 250:300] = 1

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

        # 每张图片随机选择100个图片作为测试
        count = min(100, len(gt_boxes))
        inds = np.random.choice(np.arange(len(gt_boxes)), size=count, replace=False)
        gt_boxes = gt_boxes[inds, :]
        gt_labels = gt_labels[inds]
        for j, (a, label) in enumerate(zip(gt_boxes, gt_labels)):  # per item
            if label == 5:
                xmin, ymin, xmax, ymax = a
                xc = (xmin + xmax) // 2
                yc = (ymin + ymax) // 2
                width  = 512
                height = 512

                cutout = []
                xoffset = max(0, xc - 256)
                yoffset = max(0, yc - 256)
                if xoffset + width > orig_width:
                    width = orig_width - xoffset
                if yoffset + height > orig_height:
                    height = orig_height - yoffset
                if width <= 0 or height <= 0 or width != 512 or height != 512:
                    continue
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=width, win_ysize=height)
                    cutout.append(band_data)
                cutout = np.stack(cutout, -1)  # RGB
                # cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                # im = cv2.resize(cutout, (256, 256))  # BGR

                # 在这里区分pos和neg，设定iou阈值，与gt的iou超过阈值认为正样本，反之负样本
                prefix = '%03d_%010d'%(ti,j)

                cv2.imwrite('%s/test/images/%03d_%010d.jpg' % (save_root, ti, j), cutout[:, :, ::-1])  # RGB-->BGR
                cv2.imwrite('%s/test/annotations/%03d_%010d.png' % (save_root, ti, j), mask)  # RGB-->BGR

                lines.append(prefix+'\n')

        #
        #
        # gt_boxes = torch.from_numpy(gt_boxes)
        # gt_labels = torch.from_numpy(gt_labels)
        # offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
        # print('offsets: ', offsets)
        #
        # # 在图片中随机采样一些点
        # random_indices_y = []
        # random_indices_x = []
        # for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up
        #     # sub_width = min(orig_width, big_subsize)
        #     # sub_height = min(orig_height, big_subsize)
        #     # if xoffset + sub_width > orig_width:
        #     #     sub_width = orig_width - xoffset
        #     # if yoffset + sub_height > orig_height:
        #     #     sub_height = orig_height - yoffset
        #
        #     print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
        #     img0 = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
        #     for b in range(3):
        #         band = ds.GetRasterBand(b + 1)
        #         img0[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)
        #     img0_sum = np.sum(img0, axis=2)
        #     indices_y, indices_x = np.where(img0_sum > 0)
        #     inds = np.arange(len(indices_x))
        #     np.random.shuffle(inds)
        #     count = min(256, len(inds))
        #     random_indices_y.append(indices_y[inds[:count]] + yoffset)
        #     random_indices_x.append(indices_x[inds[:count]] + xoffset)
        #
        #     del img0, img0_sum, indices_y, indices_x
        #
        # random_indices_y = np.concatenate(random_indices_y).reshape(-1, 1)
        # random_indices_x = np.concatenate(random_indices_x).reshape(-1, 1)
        # print(random_indices_y.shape, random_indices_x.shape)
        # print(random_indices_y[:10])
        # print(random_indices_x[:10])
        #
        # boxes = np.concatenate([random_indices_x - 128, random_indices_y - 128,
        #                         random_indices_x + 128, random_indices_y + 128], axis=1)
        # labels = 5 * np.ones((boxes.shape[0],))
        # boxes = torch.from_numpy(boxes)
        # labels = torch.from_numpy(labels)
        #
        # boxes = torch.cat([boxes, gt_boxes], dim=0)
        # labels = torch.cat([labels, gt_labels], dim=0)
        #
        # # 生成负样本boxes，随机采样整个图像，
        # # 这里应该随机平移和旋转，生成更多的图片
        # if True:
        #     # 提取image patches
        #     # Reshape and pad cutouts
        #     b = xyxy2xywh(boxes[:, :4])  # boxes
        #     b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
        #     b[:, 2:] = b[:, 2:] * 1.3 + 32  # pad
        #     # d[:, :4] = xywh2xyxy(b).long()
        #
        #     ious = box_iou(boxes[:, :4], gt_boxes[:, :4])
        #
        #     # import pdb
        #     # pdb.set_trace()
        #
        #     ims = []
        #     for j, (a, label) in enumerate(zip(b, labels)):  # per item
        #         if label == 5:
        #
        #             cutout = []
        #             xc, yc = int(a[0]), int(a[1])
        #             width, height = int(a[2]), int(a[3])
        #             xoffset = max(0, xc - width // 2)
        #             yoffset = max(0, yc - height // 2)
        #             if xoffset + width > orig_width:
        #                 width = orig_width - xoffset
        #             if yoffset + height > orig_height:
        #                 height = orig_height - yoffset
        #             if width <=0 or height <=0:
        #                 continue
        #             for bi in range(3):
        #                 band = ds.GetRasterBand(bi + 1)
        #                 band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=width, win_ysize=height)
        #                 cutout.append(band_data)
        #             cutout = np.stack(cutout, -1)  # RGB
        #             # cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
        #             # im = cv2.resize(cutout, (256, 256))  # BGR
        #
        #             # 在这里区分pos和neg，设定iou阈值，与gt的iou超过阈值认为正样本，反之负样本
        #             if ious[j].max() > 0.01:
        #                 save_filename = '%s/line/%03d_%010d.jpg' % (save_root, ti, j)
        #                 lines.append('%s 1\n' % save_filename.replace('E:/line_patches_gt/', ''))
        #                 prefixes.append('%03d_%010d\n'%(ti, j))
        #             else:
        #                 save_filename = '%s/nonline/%03d_%010d.jpg' % (save_root, ti, j)
        #                 lines.append('%s 0\n' % save_filename.replace('E:/line_patches_gt/', ''))
        #
        #             cv2.imwrite(save_filename, im[:, :, ::-1])  # RGB-->BGR
        #
        #             # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        #             # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
        #             # im /= 255.0  # 0 - 255 to 0.0 - 1.0
        #             # ims.append(im)

    if len(lines) > 0:
        with open(save_root + '/test.txt', 'w') as fp:
            fp.writelines(lines)
    # if len(prefixes) > 0:
    #     with open(save_root + '/%s.txt' % subset, 'w') as fp:
    #         fp.writelines(prefixes)


# 放大两倍试试，效果不好
def main_enlarge2times(subset='train'):
    source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
    gt_dir = 'F:/gddata/aerial'    # *_gt_5.xml保存在这个目录下了
    save_root = 'E:/line_patches512_for_testing_enlarge2times/%s' % (subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root + "/test2/images")
        os.makedirs(save_root + "/test2/annotations")

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

    prefixes = []

    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[240:270, 250:300] = 1

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

        # 每张图片随机选择100个图片作为测试
        count = min(100, len(gt_boxes))
        inds = np.random.choice(np.arange(len(gt_boxes)), size=count, replace=False)
        gt_boxes = gt_boxes[inds, :]
        gt_labels = gt_labels[inds]
        for j, (a, label) in enumerate(zip(gt_boxes, gt_labels)):  # per item
            if label == 5:
                xmin, ymin, xmax, ymax = a
                xc = (xmin + xmax) // 2
                yc = (ymin + ymax) // 2
                width  = 512
                height = 512

                cutout = []
                xoffset = max(0, xc - 256)
                yoffset = max(0, yc - 256)
                if xoffset + width > orig_width:
                    width = orig_width - xoffset
                if yoffset + height > orig_height:
                    height = orig_height - yoffset
                if width <= 0 or height <= 0 or width != 512 or height != 512:
                    continue
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=width, win_ysize=height)
                    cutout.append(band_data)
                cutout = np.stack(cutout, -1)  # RGB
                # cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                # im = cv2.resize(cutout, (256, 256))  # BGR

                cutout = cv2.resize(cutout, (1024, 1024))

                # 在这里区分pos和neg，设定iou阈值，与gt的iou超过阈值认为正样本，反之负样本
                prefix = '%03d_%010d'%(ti,j)

                cv2.imwrite('%s/test2/images/%03d_%010d.jpg' % (save_root, ti, j), cutout[:, :, ::-1])  # RGB-->BGR
                cv2.imwrite('%s/test2/annotations/%03d_%010d.png' % (save_root, ti, j), mask)  # RGB-->BGR

                lines.append(prefix+'\n')

        #
        #
        # gt_boxes = torch.from_numpy(gt_boxes)
        # gt_labels = torch.from_numpy(gt_labels)
        # offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
        # print('offsets: ', offsets)
        #
        # # 在图片中随机采样一些点
        # random_indices_y = []
        # random_indices_x = []
        # for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up
        #     # sub_width = min(orig_width, big_subsize)
        #     # sub_height = min(orig_height, big_subsize)
        #     # if xoffset + sub_width > orig_width:
        #     #     sub_width = orig_width - xoffset
        #     # if yoffset + sub_height > orig_height:
        #     #     sub_height = orig_height - yoffset
        #
        #     print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
        #     img0 = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
        #     for b in range(3):
        #         band = ds.GetRasterBand(b + 1)
        #         img0[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)
        #     img0_sum = np.sum(img0, axis=2)
        #     indices_y, indices_x = np.where(img0_sum > 0)
        #     inds = np.arange(len(indices_x))
        #     np.random.shuffle(inds)
        #     count = min(256, len(inds))
        #     random_indices_y.append(indices_y[inds[:count]] + yoffset)
        #     random_indices_x.append(indices_x[inds[:count]] + xoffset)
        #
        #     del img0, img0_sum, indices_y, indices_x
        #
        # random_indices_y = np.concatenate(random_indices_y).reshape(-1, 1)
        # random_indices_x = np.concatenate(random_indices_x).reshape(-1, 1)
        # print(random_indices_y.shape, random_indices_x.shape)
        # print(random_indices_y[:10])
        # print(random_indices_x[:10])
        #
        # boxes = np.concatenate([random_indices_x - 128, random_indices_y - 128,
        #                         random_indices_x + 128, random_indices_y + 128], axis=1)
        # labels = 5 * np.ones((boxes.shape[0],))
        # boxes = torch.from_numpy(boxes)
        # labels = torch.from_numpy(labels)
        #
        # boxes = torch.cat([boxes, gt_boxes], dim=0)
        # labels = torch.cat([labels, gt_labels], dim=0)
        #
        # # 生成负样本boxes，随机采样整个图像，
        # # 这里应该随机平移和旋转，生成更多的图片
        # if True:
        #     # 提取image patches
        #     # Reshape and pad cutouts
        #     b = xyxy2xywh(boxes[:, :4])  # boxes
        #     b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
        #     b[:, 2:] = b[:, 2:] * 1.3 + 32  # pad
        #     # d[:, :4] = xywh2xyxy(b).long()
        #
        #     ious = box_iou(boxes[:, :4], gt_boxes[:, :4])
        #
        #     # import pdb
        #     # pdb.set_trace()
        #
        #     ims = []
        #     for j, (a, label) in enumerate(zip(b, labels)):  # per item
        #         if label == 5:
        #
        #             cutout = []
        #             xc, yc = int(a[0]), int(a[1])
        #             width, height = int(a[2]), int(a[3])
        #             xoffset = max(0, xc - width // 2)
        #             yoffset = max(0, yc - height // 2)
        #             if xoffset + width > orig_width:
        #                 width = orig_width - xoffset
        #             if yoffset + height > orig_height:
        #                 height = orig_height - yoffset
        #             if width <=0 or height <=0:
        #                 continue
        #             for bi in range(3):
        #                 band = ds.GetRasterBand(bi + 1)
        #                 band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=width, win_ysize=height)
        #                 cutout.append(band_data)
        #             cutout = np.stack(cutout, -1)  # RGB
        #             # cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
        #             # im = cv2.resize(cutout, (256, 256))  # BGR
        #
        #             # 在这里区分pos和neg，设定iou阈值，与gt的iou超过阈值认为正样本，反之负样本
        #             if ious[j].max() > 0.01:
        #                 save_filename = '%s/line/%03d_%010d.jpg' % (save_root, ti, j)
        #                 lines.append('%s 1\n' % save_filename.replace('E:/line_patches_gt/', ''))
        #                 prefixes.append('%03d_%010d\n'%(ti, j))
        #             else:
        #                 save_filename = '%s/nonline/%03d_%010d.jpg' % (save_root, ti, j)
        #                 lines.append('%s 0\n' % save_filename.replace('E:/line_patches_gt/', ''))
        #
        #             cv2.imwrite(save_filename, im[:, :, ::-1])  # RGB-->BGR
        #
        #             # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        #             # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
        #             # im /= 255.0  # 0 - 255 to 0.0 - 1.0
        #             # ims.append(im)

    if len(lines) > 0:
        with open(save_root + '/test2.txt', 'w') as fp:
            fp.writelines(lines)
    # if len(prefixes) > 0:
    #     with open(save_root + '/%s.txt' % subset, 'w') as fp:
    #         fp.writelines(prefixes)




if __name__ == '__main__':
    subset = sys.argv[1]
    main(subset=subset)
    # main_enlarge2times(subset=subset)









