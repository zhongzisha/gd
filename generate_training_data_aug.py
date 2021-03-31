
import os
from PIL import Image
import xml.dom.minidom
import numpy as np
import cv2
import shapely.geometry as shgeo
from shapely.geometry import Point
import json
from natsort import natsorted
import math
import glob
from osgeo import gdal
import psutil
import xml.dom.minidom

# 加入旋转，大图进行旋转，box跟着变，然后


def compute_offsets(height, width, subsize, gap):

    slide = subsize - gap
    start_positions = []
    left, up = 0, 0
    while left < width:
        # if left + subsize >= width:
        #     left = max(width - subsize, 0)
        up = 0
        while up < height:
            # if up + subsize >= height:
            #     up = max(height - subsize, 0)
            right = min(left + subsize, width)
            down = min(up + subsize, height)
            sub_width = right - left
            sub_height = down - up

            start_positions.append([left, up, sub_width, sub_height])

            if up + subsize >= height:
                break
            else:
                up = up + slide
        if left + subsize >= width:
            break
        else:
            left = left + slide
    return start_positions


def calchalf_iou( poly1, poly2):
    """
        It is not the iou on usual, the iou is the value of intersection over poly1
    """
    inter_poly = poly1.intersection(poly2)
    inter_area = inter_poly.area
    poly1_area = poly1.area
    half_iou = inter_area / poly1_area
    return inter_poly, half_iou


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def GetPoly4FromPoly5(poly):
    distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1]), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i
                 in range(int(len(poly) / 2 - 1))]
    distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
    pos = np.array(distances).argsort()[0]
    count = 0
    outpoly = []
    while count < 5:
        # print('count:', count)
        if (count == pos):
            outpoly.append((poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
            outpoly.append((poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) / 2)
            count = count + 1
        elif (count == (pos + 1) % 5):
            count = count + 1
            continue

        else:
            outpoly.append(poly[count * 2])
            outpoly.append(poly[count * 2 + 1])
            count = count + 1
    return outpoly


def polyorig2sub(left, up, poly):
    polyInsub = np.zeros(len(poly))
    for i in range(int(len(poly) / 2)):
        polyInsub[i * 2] = int(poly[i * 2] - left)
        polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
    return polyInsub


def rotate_rect(base_rect):
    base_rect = base_rect.reshape([4, 2])
    # base_rect: 4x2
    base_rect = np.concatenate([base_rect,
                                np.array([1, 1, 1, 1]).reshape((4, 1))], axis=1).T  # 3x4
    base_rect[1, :] *= -1
    pc = np.mean(base_rect, axis=1)
    xc, yc = pc[0], pc[1]
    polys = []
    for angle in np.arange(0, 360, 1):
        t = angle * np.pi / 180.
        # 先将框平移到原点
        new_rect = base_rect - np.array([xc, yc, 0]).reshape((3, 1)).repeat(axis=1, repeats=4)
        M = np.array([np.cos(t), -np.sin(t), xc,
                      np.sin(t), np.cos(t), yc,
                      0, 0, 1]).reshape(3, 3)
        # 沿着原点旋转后再平移到原来的中心点
        new_rect = np.dot(M, new_rect)  # 3x4
        new_rect[1, :] *= -1
        polys.append(new_rect[:2, :].T.reshape(1, 8))


def get_gt_boxes(gt_xml_filename):
    DomTree = xml.dom.minidom.parse(gt_xml_filename)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')  # [<DOM Element: filename at 0x381f788>]
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    # print(filename, '=' * 50)

    boxes = []
    labels = []
    for objects in objectlist:
        # print objects

        namelist = objects.getElementsByTagName('name')
        if len(namelist) == 0:
            continue

        # print 'namelist:',namelist
        objectname = namelist[0].childNodes[0].data
        labels.append(int(objectname))

        bndbox = objects.getElementsByTagName('bndbox')[0]
        x1_list = bndbox.getElementsByTagName('xmin')
        x1 = int(float(x1_list[0].childNodes[0].data))
        y1_list = bndbox.getElementsByTagName('ymin')
        y1 = int(float(y1_list[0].childNodes[0].data))
        x2_list = bndbox.getElementsByTagName('xmax')
        x2 = int(float(x2_list[0].childNodes[0].data))
        y2_list = bndbox.getElementsByTagName('ymax')
        y2 = int(float(y2_list[0].childNodes[0].data))
        w = x2 - x1
        h = y2 - y1

        boxes.append([x1, y1, x2, y2])
    return boxes, labels


def load_gt(gt_xml_dir, gt_prefix, gt_subsize=5120, gt_gap=128):
    xmlfiles = glob.glob(gt_xml_dir + '/{}*.xml'.format(gt_prefix))
    all_boxes = []
    all_labels = []

    for xmlfile in xmlfiles:

        boxes, labels = get_gt_boxes(xmlfile)
        i, j = xmlfile.split(os.sep)[-1].replace('.xml', '').split('_')[1:3]
        up, left = (int(float(i)) - 1) * (gt_subsize - gt_gap), (int(float(j)) - 1) * (gt_subsize - gt_gap)
        if len(boxes) > 0:
            boxes = np.array(boxes)
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += up
            all_boxes.append(boxes)
            all_labels += labels

    all_boxes = np.concatenate(all_boxes)

    return all_boxes, all_labels


def load_gt_from_txt(filename):
    if not os.path.exists(filename):
        return [], []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    lines = np.array([line.strip().split(' ') for line in lines])
    boxes = lines[:, :4].astype(np.float32).tolist()
    labels = [int(float(val)) for val in lines[:, 4].flatten()]
    return boxes, labels


def load_gt_from_esri_xml(filename, gdal_trans_info):
    if not os.path.exists(filename):
        return [], []
    DomTree = xml.dom.minidom.parse(filename)
    annotation = DomTree.documentElement
    regionlist = annotation.getElementsByTagName('Region')
    boxes = []
    labels = []
    for region in regionlist:
        name = region.getAttribute("name")
        label = name.split('_')[0]
        polylist = region.getElementsByTagName('Coordinates')
        for poly in polylist:
            coords_str = poly.childNodes[0].data
            coords = [float(val) for val in coords_str.strip().split(' ')]
            xmin, ymin, xmax, ymax = coords[0], coords[1], coords[4], coords[5]
            x1 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
            y1 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]
            x3 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
            y3 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]
            boxes.append(np.array([x1, y1, x3, y3]))
            labels.append(int(float(label)))    # 0 is gan, 1 is jueyuanzi

    return boxes, labels


def main_test_gt():
    tif_filename = 'E:/Downloads/detect_5120_512_exp10_0/110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）.tif'
    gt_txt_filename = 'E:/Downloads/detect_5120_512_exp10_0/110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）_gt.txt'
    gt_xml_filename = 'E:/110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）_gt_new.xml'

    ds = gdal.Open(tif_filename, gdal.GA_ReadOnly)
    geotransform = ds.GetGeoTransform()

    gt_boxes1, gt_labels1 = load_gt_from_txt(gt_txt_filename)
    gt_boxes2, gt_labels2 = load_gt_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform)
    gt_boxes = gt_boxes1 + gt_boxes2
    gt_labels = gt_labels1 + gt_labels2

    print(gt_boxes)
    print(gt_labels)


def main(subset='train'):
    orig_img_path = None
    if subset == 'train':
        orig_img_path = '/media/ubuntu/Working/rs/guangdong_aerial/aerial/'
    elif subset == 'val':
        orig_img_path = '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/'
    gt_dir = '/media/ubuntu/Data/gd_gt/'
    big_subsize = 51200
    gt_gap = 128
    save_root = '/media/ubuntu/Data/gd_1024_aug/%s/' % subset
    save_img_path = '%s/images/' % save_root
    save_img_shown_path = '%s/images_shown/' % save_root
    save_txt_path = '%s/labels/' % save_root
    for p in [save_img_path, save_txt_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    tiffiles = natsorted(glob.glob(orig_img_path + '/*.tif'))
    print(tiffiles)

    list_lines = []

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["1", "2"]):  # 1 is gan, 2 is jueyuanzi
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    for i, tiffile in enumerate(tiffiles):

        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        if file_prefix != '110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）':
            continue

        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        orig_height = ds.RasterYSize
        orig_width = ds.RasterXSize
        geotransform = ds.GetGeoTransform()

        # 加载gt，分两部分，一部分是txt格式的。一部分是esri xml格式的
        gt_boxes1, gt_labels1 = load_gt_from_txt(os.path.join(gt_dir, file_prefix + '_gt.txt'))
        gt_boxes2, gt_labels2 = load_gt_from_esri_xml(os.path.join(gt_dir, file_prefix + '_gt_new.xml'),
                                                      gdal_trans_info=geotransform)
        gt_boxes = gt_boxes1 + gt_boxes2
        gt_labels = gt_labels1 + gt_labels2
        all_boxes = np.concatenate([np.array(gt_boxes, dtype=np.float32).reshape(-1, 4),
                                    np.array(gt_labels, dtype=np.float32).reshape(-1, 1)], axis=1)
        print('all_boxes')
        print(all_boxes)

        # 先计算可用内存，如果可以放得下，就不用分块了
        avaialble_mem_bytes = psutil.virtual_memory().available
        if orig_width * orig_height * ds.RasterCount < 0.8 * avaialble_mem_bytes:
            offsets = [[0, 0, orig_width, orig_height]]
        else:
            # 根据big_subsize计算子块的起始偏移
            big_subsize = int(np.sqrt(0.8 * avaialble_mem_bytes / 3) - gt_gap)
            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
        print('offsets: ', offsets)

        for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up

            print(oi, '=' * 50)
            print('offset = ', (xoffset, yoffset))
            print('size = ', (sub_width, sub_height))
            print('reading data to img')
            img0 = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
            for b in range(3):
                band = ds.GetRasterBand(b + 1)
                img0[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)

            boxes = np.copy(gt_boxes)
            labels = np.copy(gt_labels)
            all_boxes_sub = np.copy(all_boxes)

            boxes[:, [0, 2]] -= xoffset
            boxes[:, [1, 3]] -= yoffset

            print('begin aug')
            height, width = img0.shape[:2]
            for bi, (box, label) in enumerate(zip(boxes, labels)):
                if label == 1:
                    xc = (box[0] + box[2]) / 2.
                    yc = (box[1] + box[3]) / 2.

                    xmin = max(0, xc - 512)
                    ymin = max(0, yc - 512)
                    xmax = xmin + 1024
                    ymax = ymin + 1024
                    if xmax >= width - 1:
                        xmax = width - 1
                        xmin = xmax - 1024
                    if ymax >= height:
                        ymax = height - 1
                        ymin = ymax - 1024
                    xmin = max(xmin, 0)
                    ymin = max(ymin, 0)

                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    w = xmax - xmin
                    h = ymax - ymin

                    im_sub = np.zeros((1024, 1024, 3), dtype=np.uint8)
                    print('(%d, %d) --> (%d, %d), (%d, %d)' % (xmin, ymin, xmax, ymax, w, h))
                    im_sub[:h, :w, :] = img0[ymin:ymax, xmin:xmax, :].copy()

                    imgpoly = shgeo.Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax),
                                             (xmin, ymax)])

                    height_sub, width_sub = im_sub.shape[:2]
                    valid_boxes = []
                    valid_lines = []
                    for obj in all_boxes_sub:
                        xmin1, ymin1, xmax1, ymax1, label = obj
                        gtpoly = shgeo.Polygon([(xmin1, ymin1),
                                                (xmax1, ymin1),
                                                (xmax1, ymax1),
                                                (xmin1, ymax1)])

                        inter_poly, half_iou = calchalf_iou(gtpoly, imgpoly)

                        if half_iou > 0.5:
                            xmin1_new = max(xmin1 - xmin, 0)
                            ymin1_new = max(ymin1 - ymin, 0)
                            xmax1_new = min(xmax1 - xmin, w - 1)
                            ymax1_new = min(ymax1 - ymin, h - 1)
                            valid_boxes.append([xmin1_new, ymin1_new, xmax1_new, ymax1_new, label])
                            xc1 = (xmin1_new + xmax1_new) / 2.
                            yc1 = (ymin1_new + ymax1_new) / 2.
                            w1 = xmax1_new - xmin1_new
                            h1 = ymax1_new - ymin1_new
                            valid_lines.append(
                                "%d %f %f %f %f\n" % (label - 1, xc1 / 1024, yc1 / 1024, w1 / 1024, h1 / 1024))

                            # for coco format
                            single_obj = {'area': int(w1 * h1),
                                          'category_id': int(label),
                                          'segmentation': []}
                            single_obj['segmentation'].append(
                                [int(xmin1_new), int(ymin1_new), int(xmax1_new), int(ymin1_new),
                                 int(xmax1_new), int(ymax1_new), int(xmin1_new), int(ymax1_new)]
                            )
                            single_obj['iscrowd'] = 0

                            single_obj['bbox'] = int(xmin1_new), int(ymin1_new), int(w1), int(h1)
                            single_obj['image_id'] = image_id
                            single_obj['id'] = inst_count
                            data_dict['annotations'].append(single_obj)
                            inst_count = inst_count + 1

                    if len(valid_boxes):

                        save_prefix = '%d_%d_%d' % (i, oi, bi)

                        if True:
                            im_sub1 = im_sub.copy()
                            for obj in valid_boxes:
                                xmin1, ymin1, xmax1, ymax1, label = obj
                                cv2.rectangle(im_sub1, (int(xmin1), int(ymin1)), (int(xmax1), int(ymax1)),
                                              color=(0, 0, 255) if label == 1 else (0, 255, 0), thickness=2,
                                              lineType=2)
                            cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', im_sub1)

                        # for coco format
                        single_image = {}
                        single_image['file_name'] = save_prefix + '.png'
                        single_image['id'] = image_id
                        single_image['width'] = width_sub
                        single_image['height'] = height_sub
                        data_dict['images'].append(single_image)
                        image_id = image_id + 1

                        # for yolo format
                        cv2.imwrite(save_img_path + save_prefix + '.png', im_sub)
                        with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                            fp.writelines(valid_lines)

                        list_lines.append('./images/%s.png\n' % save_prefix)

    if len(list_lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_root + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


def main1(subset='train'):
    subsize = 5120
    gap = 128
    if subset == 'train':
        orig_img_path = '/media/ubuntu/Temp/gd/data/aerial/%d_%d/' % (subsize, gap)
    elif subset == 'val':
        orig_img_path = '/media/ubuntu/Temp/gd/data/aerial2/%d_%d/' % (subsize, gap)
    anno_path = '/home/ubuntu/Downloads/Annotations/%s/'%subset
    save_root = '/media/ubuntu/Data/gd_1024/%s/'%subset
    save_img_path = '%s/images/' % save_root
    save_img_shown_path = '%s/images_shown/' % save_root
    save_txt_path = '%s/labels/' % save_root
    for p in [save_img_path, save_txt_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)
    xmlfiles = natsorted(os.listdir(anno_path))

    list_lines = []

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["1", "2"]):  # 1 is gan, 2 is jueyuanzi
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    for xmlfile in xmlfiles:

        # if image_id == 5:
        #     break

        DomTree = xml.dom.minidom.parse(anno_path + xmlfile)
        annotation = DomTree.documentElement

        filenamelist = annotation.getElementsByTagName('filename')  # [<DOM Element: filename at 0x381f788>]
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        print(filename, '='*50)

        boxes = []
        labels = []
        for objects in objectlist:
            # print objects

            namelist = objects.getElementsByTagName('name')
            if len(namelist) == 0:
                continue

            # print 'namelist:',namelist
            objectname = namelist[0].childNodes[0].data
            labels.append(int(objectname))

            bndbox = objects.getElementsByTagName('bndbox')[0]
            x1_list = bndbox.getElementsByTagName('xmin')
            x1 = int(float(x1_list[0].childNodes[0].data))
            y1_list = bndbox.getElementsByTagName('ymin')
            y1 = int(float(y1_list[0].childNodes[0].data))
            x2_list = bndbox.getElementsByTagName('xmax')
            x2 = int(float(x2_list[0].childNodes[0].data))
            y2_list = bndbox.getElementsByTagName('ymax')
            y2 = int(float(y2_list[0].childNodes[0].data))
            w = x2 - x1
            h = y2 - y1

            boxes.append([x1, y1, x2, y2])

        print(boxes)
        print(labels)

        folder_name = filename.split('_')[0]
        orig_img_filename = os.path.join(orig_img_path, folder_name, filename)
        im = cv2.imread(orig_img_filename)

        all_boxes = np.concatenate([np.array(boxes, dtype=np.float32).reshape(-1, 4),
                                    np.array(labels, dtype=np.float32).reshape(-1, 1)], axis=1)

        height, width = im.shape[:2]
        for bi, (box, label) in enumerate(zip(boxes, labels)):
            if label == 1:
                xc = (box[0] + box[2])/2.
                yc = (box[1] + box[3])/2.

                xmin = max(0, xc - 512)
                ymin = max(0, yc - 512)
                xmax = xmin + 1024
                ymax = ymin + 1024
                if xmax >= width-1:
                    xmax = width-1
                    xmin = xmax - 1024
                if ymax >= height:
                    ymax = height-1
                    ymin = ymax - 1024
                xmin = max(xmin, 0)
                ymin = max(ymin, 0)

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                w = xmax - xmin
                h = ymax - ymin

                im_sub = np.zeros((1024, 1024, 3), dtype=np.uint8)
                print('(%d, %d) --> (%d, %d), (%d, %d)' % (xmin, ymin, xmax, ymax, w, h))
                im_sub[:h, :w, :] = im[ymin:ymax, xmin:xmax, :].copy()

                imgpoly = shgeo.Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax),
                                         (xmin, ymax)])

                height_sub, width_sub = im_sub.shape[:2]
                valid_boxes = []
                valid_lines = []
                for obj in all_boxes:
                    xmin1, ymin1, xmax1, ymax1, label = obj
                    gtpoly = shgeo.Polygon([(xmin1, ymin1),
                                            (xmax1, ymin1),
                                            (xmax1, ymax1),
                                            (xmin1, ymax1)])

                    inter_poly, half_iou = calchalf_iou(gtpoly, imgpoly)

                    if half_iou > 0.5:
                        xmin1_new = max(xmin1 - xmin, 0)
                        ymin1_new = max(ymin1 - ymin, 0)
                        xmax1_new = min(xmax1 - xmin, w-1)
                        ymax1_new = min(ymax1 - ymin, h-1)
                        valid_boxes.append([xmin1_new, ymin1_new, xmax1_new, ymax1_new, label])
                        xc1 = (xmin1_new + xmax1_new) / 2.
                        yc1 = (ymin1_new + ymax1_new) / 2.
                        w1 = xmax1_new - xmin1_new
                        h1 = ymax1_new - ymin1_new
                        valid_lines.append("%d %f %f %f %f\n" % (label-1, xc1/1024, yc1/1024, w1/1024, h1/1024))

                        # for coco format
                        single_obj = {'area': int(w1 * h1),
                                      'category_id': int(label),
                                      'segmentation': []}
                        single_obj['segmentation'].append(
                            [int(xmin1_new), int(ymin1_new), int(xmax1_new), int(ymin1_new),
                             int(xmax1_new), int(ymax1_new), int(xmin1_new), int(ymax1_new)]
                        )
                        single_obj['iscrowd'] = 0

                        single_obj['bbox'] = int(xmin1_new), int(ymin1_new), int(w1), int(h1)
                        single_obj['image_id'] = image_id
                        single_obj['id'] = inst_count
                        data_dict['annotations'].append(single_obj)
                        inst_count = inst_count + 1

                if len(valid_boxes):

                    save_prefix = '%s_%d' % (filename.replace('.tif', ''), bi)

                    if True:
                        im_sub1 = im_sub.copy()
                        for obj in valid_boxes:
                            xmin1, ymin1, xmax1, ymax1, label = obj
                            cv2.rectangle(im_sub1, (int(xmin1), int(ymin1)), (int(xmax1), int(ymax1)),
                                          color=(0,0,255) if label == 1 else (0, 255, 0), thickness=2,
                                          lineType=2)
                        cv2.imwrite(save_img_shown_path + filename.replace('.tif', '_shown_%d.png' % bi), im_sub1)

                    # for coco format
                    single_image = {}
                    single_image['file_name'] = save_prefix + '.png'
                    single_image['id'] = image_id
                    single_image['width'] = width_sub
                    single_image['height'] = height_sub
                    data_dict['images'].append(single_image)
                    image_id = image_id + 1

                    # for yolo format
                    cv2.imwrite(save_img_path + save_prefix + '.png', im_sub)
                    with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                        fp.writelines(valid_lines)

                    list_lines.append('./images/%s.png\n' % save_prefix)

    if len(list_lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_root + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


if __name__ == '__main__':
    # main(subset='val')
    # main_test_gt()
    main(subset='train')