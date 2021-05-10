import sys, os, glob

import cv2
import numpy as np
import torch
from osgeo import gdal, osr
from natsort import natsorted
from yoloV5.myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, box_iou_np, \
    box_intersection_np
import json
import socket

"""
从gt中提取图像块，包含gt boxes
"""


def load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info, valid_labels):
    print(gt_txt_filename)
    print(gt_xml_filename)
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
    print('all_boxes')
    print(all_boxes)

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


def main(subset='train', aug_times=1, save_img=False,
         save_root=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    save_root = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    gt_postfix = '_gt_5.xml'
    valid_labels_set = [1, 2, 3, 4]

    save_img_path = '%s/images/' % save_root
    save_img_shown_path = '%s/images_shown/' % save_root
    save_txt_path = '%s/labels/' % save_root
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

    colors = {1:(255, 0, 0), 2:(0, 255, 0), 3:(0, 0, 255), 4:(255, 255, 0)}

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
                    box_w, box_h = int(xmax - xmin), int(ymax-ymin)

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
                                    sub_gt_boxes = [box3 for ii, box3 in enumerate(sub_gt_boxes) if ii not in idx2.tolist()]

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
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_root + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('python ./this_script.py aug_times(int) save_img(bool)')
        sys.exit(-1)

    aug_times = int(sys.argv[1])
    save_img = int(sys.argv[2]) != 0

    hostname = socket.gethostname()
    if hostname == 'master':
        save_root = '/media/ubuntu/Data/gd_newAug%d_4classes' % aug_times
    else:
        save_root = 'E:/gd_newAug%d_4classes' % aug_times

    main(subset='train', aug_times=aug_times, save_img=save_img, save_root=save_root)
    main(subset='val', aug_times=2, save_img=save_img, save_root=save_root)
