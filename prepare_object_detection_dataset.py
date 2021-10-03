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

from xpinyin import Pinyin


"""
pip install xpinyin -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
"""

def prepare_object_detection_dataset(save_root):
    hostname = socket.gethostname()
    source = 'E:/all_tif_files.txt'

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    # tiffiles = [r'G:\gddata\all\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif']

    for tiffile in tiffiles:
        process_one_tif(save_root, tiffile)


# random points according to the ground truth polygons
def process_one_tif(save_root=None, tiffile=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        gt_dir = 'G:/gddata/all'  # sys.argv[2]

    pinyin = Pinyin()
    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
    print(file_prefix)
    if '（' in file_prefix and '）' in file_prefix:
        invalid_0 = file_prefix.find('（')
        invalid_1 = file_prefix.find('）')
        new_file_prefix = file_prefix[:invalid_0] + file_prefix[(invalid_1+1):]
        new_file_prefix = new_file_prefix.replace('杆塔、导线、绝缘子、树木', '')
    else:
        new_file_prefix = file_prefix
    new_file_prefix = pinyin.get_pinyin(new_file_prefix.replace('、', '_'))

    save_dir = '%s/%s/' % (save_root, new_file_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        return

    # valid_labels_set = [1, 2, 3, 4]
    valid_labels_set = [3, 4]

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

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

    lines = []
    size0 = 10000
    size1 = -1

    subsizes = [1024]
    scales = [1.0]
    gap = 128

    if True:

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
        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                    valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            return

        for si, (subsize, scale) in enumerate(zip(subsizes, scales)):

            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=subsize, gap=gap)

            for oi, (xoffset, yoffset, sub_w, sub_h) in enumerate(offsets):  # left, up
                # sub_width = min(orig_width, big_subsize)
                # sub_height = min(orig_height, big_subsize)
                # if xoffset + sub_width > orig_width:
                #     sub_width = orig_width - xoffset
                # if yoffset + sub_height > orig_height:
                #     sub_height = orig_height - yoffset
                print(oi, len(offsets), xoffset, yoffset, sub_w, sub_h)
                save_prefix = '%s_%d_%d' % (new_file_prefix, si, oi)

                xoffset = max(1, xoffset)
                yoffset = max(1, yoffset)
                if xoffset + sub_w > orig_width - 1:
                    sub_w = orig_width - 1 - xoffset
                if yoffset + sub_h > orig_height - 1:
                    sub_h = orig_height - 1 - yoffset
                xoffset, yoffset, sub_w, sub_h = [int(val) for val in
                                                  [xoffset, yoffset, sub_w, sub_h]]
                xmin1, ymin1 = xoffset, yoffset
                xmax1, ymax1 = xoffset + sub_w, yoffset + sub_h

                cutout = []
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_w, win_ysize=sub_h)
                    cutout.append(band_data)
                cutout = np.stack(cutout, -1)  # RGB
                # cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                # im = cv2.resize(cutout, (256, 256))  # BGR
                # cv2.imwrite(save_filename, im)  # 不能有中文

                if np.min(cutout[:, :, 0]) == np.max(cutout[:, :, 0]):
                    continue

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

                # if oi == 277:
                #     import pdb
                #     pdb.set_trace()
                #
                # for ib, box in enumerate(sub_gt_boxes):
                #     print('sub_gt', ib, box)
                # if len(np.where(im1[:, :, 0] > 0)[0]) < 0.5 * np.prod(im1.shape[:2]):
                #     continue

                if len(invalid_gt_boxes) > 0:
                    for box2 in np.array(invalid_gt_boxes).astype(np.int32):
                        xmin, ymin, xmax, ymax, label = box2
                        cutout[ymin:ymax, xmin:xmax, :] = 0

                        # check the sub_gt_boxes, remove those boxes where in invalid_gt_boxes
                        if len(sub_gt_boxes) > 0:
                            # ious = box_iou_np(box2[:4].reshape(1, 4),
                            #                   np.array(sub_gt_boxes, dtype=np.float32).reshape(-1, 5)[:, :4])
                            # idx2 = np.where(ious > 0)[1]
                            # if len(idx2) > 0:
                            #     sub_gt_boxes = [box3 for ii, box3 in enumerate(sub_gt_boxes) if
                            #                     ii not in idx2.tolist()]
                            def box_in_box(tmpbox):
                                xmin1, ymin1, xmax1, ymax1, label = tmpbox
                                if xmin <= xmin1 <= xmax and ymin <= ymin1 <= ymax and \
                                    xmin <= xmax1 <= xmax and ymin <= ymax1 <= ymax:
                                    return True
                                else:
                                    return False

                            sub_gt_boxes = [box3 for ii, box3 in enumerate(sub_gt_boxes) if not box_in_box(box3)]

                # if oi == 277:
                #     import pdb
                #     pdb.set_trace()
                # draw gt boxes
                if len(sub_gt_boxes) > 0 and np.any(np.array(sub_gt_boxes)[:, -1] == 3):

                    # save image
                    # for coco format
                    single_image = {}
                    single_image['file_name'] = save_prefix + '.jpg'
                    single_image['id'] = image_id
                    single_image['width'] = sub_w
                    single_image['height'] = sub_h
                    data_dict['images'].append(single_image)

                    # for yolo format
                    # cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout)  # RGB --> BGR
                    Image.fromarray(cutout).save(save_img_path + save_prefix + '.jpg')

                    list_lines.append('./images/%s.jpg\n' % save_prefix)

                    save_img = True

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
                        # cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout)  # RGB --> BGR
                        Image.fromarray(cutout).save(save_img_shown_path + save_prefix + '.jpg')
                else:
                    # if no gt_boxes
                    if len(np.where(cutout[:, :, 0] > 0)[0]) < 0.5 * np.prod(cutout.shape[:2]):
                        continue

                    if np.random.rand() < 0.001:
                        # for yolo format
                        save_prefix = save_prefix + '_noGT'
                        # cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout)  # RGB --> BGR
                        Image.fromarray(cutout).save(save_img_path + save_prefix + '.jpg')

                        list_lines.append('./images/%s.jpg\n' % save_prefix)

                        with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                            pass

                # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                # ims.append(im)

    if len(list_lines) > 0:
        with open(save_dir + '/gt.txt', 'w') as fp:
            fp.writelines(list_lines)

        with open(save_dir + '/gt.json', 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


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
        save_root = '/media/ubuntu/Data/gd_newAug%d_Rot%d_4classes_object_detection' % (aug_times, do_rotate)
    elif hostname == 'master3':
        save_root = '/media/ubuntu/Temp/gd_newAug%d_Rot%d_4classes_object_detection' % (aug_times, do_rotate)
    else:
        save_root = r'E:\gd_newAug%d_Rot%d_4classes_object_detection_test' % (aug_times, do_rotate)

    if aug_type == 'object_detection':
        save_root = '%s/%s' % (save_root, aug_type)
        prepare_object_detection_dataset(save_root=save_root)
        sys.exit(-1)
