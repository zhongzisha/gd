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


def calchalf_iou(poly1, poly2):
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


def rotate_rect(base_rect, angle_step=30):
    base_rect = base_rect.reshape([4, 2])
    # base_rect: 4x2
    base_rect = np.concatenate([base_rect,
                                np.array([1, 1, 1, 1]).reshape((4, 1))], axis=1).T  # 3x4
    base_rect[1, :] *= -1
    pc = np.mean(base_rect, axis=1)
    xc, yc = pc[0], pc[1]
    polys = []
    for angle in np.arange(0, 360, angle_step):
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

    return polys


def main_test_rotate():
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    base_rect = np.array([200, 200, 500, 200, 500, 500, 200, 500], dtype=np.float32)
    polys = rotate_rect(base_rect)

    for i, poly in enumerate(polys):
        quad = poly.reshape([4, 2]).astype(np.int32)
        cv2.drawContours(img, [quad], -1, color=(0, 255, 0), thickness=2)
        cv2.putText(img, text=str(i), org=(int(quad[0, 0]), int(quad[0, 1])), fontFace=1, fontScale=1,
                    color=(0, 255, 255), thickness=1)

    cv2.imwrite('test_rotate.jpg', img)


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
    xOrigin = gdal_trans_info[0]
    yOrigin = gdal_trans_info[3]
    pixelWidth = gdal_trans_info[1]
    pixelHeight = gdal_trans_info[5]

    for region in regionlist:
        name = region.getAttribute("name")
        label = name.split('_')[0]
        polylist = region.getElementsByTagName('Coordinates')
        for poly in polylist:
            coords_str = poly.childNodes[0].data
            coords = [float(val) for val in coords_str.strip().split(' ')]
            xmin, ymin, xmax, ymax = coords[0], coords[1], coords[4], coords[5]   # this is the map coordinates
            # x1 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
            # y1 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]
            # x3 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
            # y3 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]
            x1 = int((xmin - xOrigin) / pixelWidth + 0.5)
            y1 = int((ymin - yOrigin) / pixelHeight + 0.5)
            x3 = int((xmax - xOrigin) / pixelWidth + 0.5)
            y3 = int((ymax - yOrigin) / pixelHeight + 0.5)

            boxes.append(np.array([x1, y1, x3, y3]))
            labels.append(int(float(label)))  # 0 is gan, 1 is jueyuanzi

    return boxes, labels


def main_test_gt():

    gt_xml_dir = '/media/ubuntu/Data/gd_gt/'
    tiffiles = natsorted(glob.glob('/media/ubuntu/Working/rs/guangdong_aerial/aerial2/*.tif'))
    print(tiffiles)
    for i, tiffile in enumerate(tiffiles):
        print(i, tiffile)
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif','')
        gt_txt_filename = os.path.join(gt_xml_dir, file_prefix+'_gt.txt')
        gt_xml_filename = os.path.join(gt_xml_dir, file_prefix+'_gt_new.xml')

        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        geotransform = ds.GetGeoTransform()
        print('geotransform', geotransform)

        gt_boxes1, gt_labels1 = load_gt_from_txt(gt_txt_filename)
        gt_boxes2, gt_labels2 = load_gt_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform)
        gt_boxes = gt_boxes1 + gt_boxes2
        gt_labels = gt_labels1 + gt_labels2

        # print(gt_boxes)
        # print(gt_labels)


def py_cpu_nms(dets, thresh):
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]

    # 这边的keep用于存放，NMS后剩余的方框
    keep = []

    # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    index = scores.argsort()[::-1]
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。

    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # every time the first is the biggst, and add it directly

        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)

        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h

        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx + 1]  # because index start from 1

    return keep


def mainaaa(subset='train', do_aug=False):
    all_list = [
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv苏程线N3-N17（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv苏隆线N3-N10（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv莱金线N26-N33_N38-N39_N17-N18（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann31-n36.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann66-n68.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann74-n82.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann39-n42.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann53-n541.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann64-n65.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann70-n71.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kv厂梅线13-14（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kv长顺线N51-N55_0.05m_杆塔、导线、绝缘子、树木.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/云南玉溪（杆塔、导线、树木、水体）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/候村250m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/威华300m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/工业园350m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/水口300m_mosaic.tif'
    ]
    train_list = [
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv苏程线N3-N17（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv莱金线N26-N33_N38-N39_N17-N18（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann31-n36.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann66-n68.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann39-n42.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann64-n65.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/威华300m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kv厂梅线13-14（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kv长顺线N51-N55_0.05m_杆塔、导线、绝缘子、树木.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/云南玉溪（杆塔、导线、树木、水体）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/工业园350m_mosaic.tif',
    ]
    val_list = [
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/候村250m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/水口300m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann74-n82.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann70-n71.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv苏隆线N3-N10（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann53-n541.tif',
    ]

    # orig_img_path = None
    # if subset == 'train':
    #     orig_img_path = '/media/ubuntu/Working/rs/guangdong_aerial/aerial/'
    # elif subset == 'val':
    #     orig_img_path = '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/'
    gt_dir = '/media/ubuntu/Data/gd_gt/'
    gt_gap = 128
    if subset == 'train':
        aug_times = 2
    else:
        aug_times = 1
    save_root = '/media/ubuntu/Data/gd_1024_aug_90_newSplit/%s/' % subset
    tiffiles = None
    if subset == 'train':
        tiffiles = natsorted(train_list)
    elif subset == 'val':
        tiffiles = natsorted(val_list)
    print(tiffiles)
    with open('%s/%s_list.txt' % (save_root, subset), 'w') as fp:
        fp.writelines([line+'\n' for line in tiffiles])


def main(subset='train', do_aug=False):
    all_list = [
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv苏程线N3-N17（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv苏隆线N3-N10（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv莱金线N26-N33_N38-N39_N17-N18（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann31-n36.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann66-n68.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann74-n82.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann39-n42.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann53-n541.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann64-n65.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann70-n71.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kv厂梅线13-14（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kv长顺线N51-N55_0.05m_杆塔、导线、绝缘子、树木.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/云南玉溪（杆塔、导线、树木、水体）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/候村250m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/威华300m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/工业园350m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/水口300m_mosaic.tif'
    ]
    train_list = [
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv苏程线N3-N17（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv莱金线N26-N33_N38-N39_N17-N18（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann31-n36.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann66-n68.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann39-n42.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann64-n65.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/威华300m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kv厂梅线13-14（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kv长顺线N51-N55_0.05m_杆塔、导线、绝缘子、树木.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/云南玉溪（杆塔、导线、树木、水体）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/工业园350m_mosaic.tif',
    ]
    val_list = [
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/候村250m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/水口300m_mosaic.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann74-n82.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann70-n71.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/110kv苏隆线N3-N10（杆塔、导线、绝缘子、树木）.tif',
        '/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvqinshunxiann53-n541.tif',
    ]

    # orig_img_path = None
    # if subset == 'train':
    #     orig_img_path = '/media/ubuntu/Working/rs/guangdong_aerial/aerial/'
    # elif subset == 'val':
    #     orig_img_path = '/media/ubuntu/Working/rs/guangdong_aerial/aerial2/'
    gt_dir = '/media/ubuntu/Data/gd_gt/'
    gt_gap = 128
    if subset == 'train':
        aug_times = 2
    else:
        aug_times = 1
    save_root = '/media/ubuntu/Data/gd_1024_aug_90_newSplit/%s/' % subset
    save_img_path = '%s/images/' % save_root
    save_img_shown_path = '%s/images_shown/' % save_root
    save_txt_path = '%s/labels/' % save_root
    for p in [save_img_path, save_txt_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    tiffiles = None
    if subset == 'train':
        tiffiles = natsorted(train_list)
    elif subset == 'val':
        tiffiles = natsorted(val_list)
    print(tiffiles)

    with open('%s/%s_list.txt' % (save_root, subset), 'w') as fp:
        fp.writelines([line+'\n' for line in tiffiles])

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

    for i in range(len(tiffiles)):
        tiffile = tiffiles[i]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        # if file_prefix != '110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）':
        #     continue

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

        # 每个类进行nms
        tmp_boxes = []
        tmp_labels = []
        for label in [1, 2]:
            idx = np.where(all_boxes[:, 4] == label)[0]
            if len(idx) > 0:
                boxes_thisclass = all_boxes[idx, :4]
                labels_thisclass = all_boxes[idx, 4:5]
                dets = np.concatenate([boxes_thisclass.astype(np.float32),
                                       0.99 * np.ones_like(idx, dtype=np.float32).reshape([-1, 1])], axis=1)
                keep = py_cpu_nms(dets, thresh=0.5)
                tmp_boxes.append(boxes_thisclass[keep])
                tmp_labels.append(labels_thisclass[keep])
        gt_boxes = np.concatenate(tmp_boxes)
        gt_labels = np.concatenate(tmp_labels)

        if len(gt_boxes) == 0 or len(gt_boxes) != len(gt_labels):
            continue

        # 先计算可用内存，如果可以放得下，就不用分块了
        avaialble_mem_bytes = psutil.virtual_memory().available
        if False:  # orig_width * orig_height * ds.RasterCount < 0.8 * avaialble_mem_bytes:
            offsets = [[0, 0, orig_width, orig_height]]
        else:
            # 根据big_subsize计算子块的起始偏移
            big_subsize = int(np.sqrt(0.8 * avaialble_mem_bytes / 3) - gt_gap)
            big_subsize = 10240
            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=gt_gap)
        print('offsets: ', offsets)

        all2_image_path = []
        for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up

            print(oi, '=' * 50)
            print('offset = ', (xoffset, yoffset))
            print('size = ', (sub_width, sub_height))
            boxes = np.copy(gt_boxes)
            labels = np.copy(gt_labels)

            boxes[:, [0, 2]] -= xoffset
            boxes[:, [1, 3]] -= yoffset

            # 只保留在截取区域内部的框
            boxes_new = []
            labels_new = []
            for bi, (box, label) in enumerate(zip(boxes, labels)):
                xmin, ymin, xmax, ymax = box
                if 0 <= xmin < xmax < sub_width and 0 <= ymin < ymax < sub_height:
                    boxes_new.append(box)
                    labels_new.append(label)
            boxes = boxes_new
            labels = labels_new

            has_gt_boxes = True
            if len(boxes) == 0 or len(boxes) != len(labels):
                # 表示当前的image patch里面没有框
                has_gt_boxes = False
                print('NO gt boxes in this region')
                # continue
            else:
                print('gt boxes in this region')

            if subset == 'val' and not has_gt_boxes:
                continue

            print('reading data to img')
            img0 = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
            for b in range(3):
                band = ds.GetRasterBand(b + 1)
                img0[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)

            save_prefix = '%d_%d_xoff%d_yoff%d' % (i, oi, xoffset, yoffset)
            im0_show = np.copy(img0)
            minval = im0_show.min()
            maxval = im0_show.max()
            all_black = False
            if maxval > minval:
                for (box, label) in zip(boxes, labels):
                    xmin1, ymin1, xmax1, ymax1 = box
                    cv2.rectangle(im0_show, (int(xmin1), int(ymin1)), (int(xmax1), int(ymax1)),
                                  color=(0, 0, 255) if label == 1 else (0, 255, 0), thickness=5,
                                  lineType=1)
                cv2.imwrite(save_img_shown_path + save_prefix + '_big.jpg', im0_show[:, :, ::-1])
            else:
                # 这里表示当前的image patch里面全为黑，也就不可能有框了
                all_black = True
                # continue

            if subset == 'val' and all_black:
                continue

            nogt_count = 0
            if subset == 'train' and (not has_gt_boxes) and (not all_black):
                height, width = img0.shape[:2]
                # 这里表示当前image patch没有标注框，且不是全黑，表示当前图像patch可以作为背景
                # 在这个背景图片里进行随机采样，扩充数据集
                img0_area = np.prod(img0.shape[:2])
                if np.sum(img0[:, :, 0] > 0) > 0.5 * img0_area:
                    rows, cols = np.where(img0[:, :, 0] > 0)
                    row_indices = np.random.choice(rows, min(1000, len(rows)), replace=False)
                    col_indices = np.random.choice(cols, min(1000, len(rows)), replace=False)
                    for nogti in range(len(row_indices)):
                        xc, yc = row_indices[nogti], col_indices[nogti]
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
                        tmpim = img0[ymin:ymax, xmin:xmax, :]
                        tmpim_area = np.prod(tmpim.shape[:2])
                        if np.sum(tmpim[:, :, 0] > 0) > 0.5 * tmpim_area:
                            save_prefix = '%d_%d_xoff%d_yoff%d_noGT%d' % (i, oi, xoffset, yoffset, nogt_count)
                            cv2.imwrite(save_img_path + save_prefix + '.jpg', tmpim[:, :, ::-1])  # RGB --> BGR
                            with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                                pass
                            list_lines.append('./images/%s.jpg\n' % save_prefix)
                            nogt_count += 1

                        if nogt_count == 10:
                            break
                continue

            if subset == 'train' and (not all_black):
                height, width = img0.shape[:2]
                boxes_copy = np.array(boxes)
                # 如果不是全黑，全图范围内随机选200个图像块，从中选取一些不包含gt_boxes的作为背景
                img0_area = np.prod(img0.shape[:2])
                if np.sum(img0[:, :, 0] > 0) > 0.5 * img0_area:
                    rows, cols = np.where(img0[:, :, 0] > 0)
                    row_indices = np.random.choice(rows, min(1000, len(rows)), replace=False)
                    col_indices = np.random.choice(cols, min(1000, len(rows)), replace=False)
                    for nogti in range(len(row_indices)):
                        xc, yc = row_indices[nogti], col_indices[nogti]
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

                        # 检查这个范围内是否有gtboxes
                        boxes_xmin = np.max(boxes_copy[:, 0])
                        boxes_ymin = np.max(boxes_copy[:, 1])
                        boxes_xmax = np.min(boxes_copy[:, 2])
                        boxes_ymax = np.min(boxes_copy[:, 3])

                        if (xmin <= boxes_xmin <= xmax and ymin <= boxes_ymin <= ymax) or \
                                (xmin <= boxes_xmax <= xmax and ymin <= boxes_ymax <= ymax):
                            continue

                        tmpim = img0[ymin:ymax, xmin:xmax, :]
                        tmpim_area = np.prod(tmpim.shape[:2])
                        if np.sum(tmpim[:, :, 0] > 0) > 0.5 * tmpim_area:
                            save_prefix = '%d_%d_xoff%d_yoff%d_noGT%d' % (i, oi, xoffset, yoffset, nogt_count)
                            cv2.imwrite(save_img_path + save_prefix + '.jpg', tmpim[:, :, ::-1])  # RGB --> BGR
                            with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                                pass
                            list_lines.append('./images/%s.jpg\n' % save_prefix)
                            nogt_count += 1

                        if nogt_count >= np.random.randint(15, 20):
                            break

            if not has_gt_boxes:
                print('no gt boxes in this region')
                continue
            if all_black:
                print('the image patch is invalid')
                continue

            all_boxes_sub = np.concatenate([np.array(boxes, dtype=np.float32).reshape(-1, 4),
                                            np.array(labels, dtype=np.float32).reshape(-1, 1)], axis=1)
            for augi in range(aug_times):
                print('begin aug time ', augi)
                height, width = img0.shape[:2]
                for bi, (box, label) in enumerate(zip(boxes, labels)):
                    if label == 1:
                        xc = (box[0] + box[2]) / 2.
                        yc = (box[1] + box[3]) / 2.

                        if subset == 'train':
                            random_shifts = np.random.randint(-256, 256, size=2)
                            xshift = random_shifts[0]
                            yshift = random_shifts[1]
                            xc += xshift
                            yc += yshift

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

                        if w == 0 or h == 0:
                            continue
                        if not (xmax > xmin and ymax > ymin):
                            continue

                        im_sub = np.zeros((1024, 1024, 3), dtype=np.uint8)
                        # print('(%d, %d) --> (%d, %d), (%d, %d)' % (xmin, ymin, xmax, ymax, w, h))
                        im_sub[:h, :w, :] = img0[ymin:ymax, xmin:xmax, :].copy()

                        imgpoly = shgeo.Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax),
                                                 (xmin, ymax)])

                        height_sub, width_sub = im_sub.shape[:2]
                        valid_boxes = []
                        valid_lines = []
                        valid_indices = []
                        for vi, obj in enumerate(all_boxes_sub.copy()):
                            xmin1, ymin1, xmax1, ymax1, label = obj
                            if not (xmax1 > xmin1 and ymax1 > ymin1):
                                continue

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
                                valid_indices.append(vi)

                        has_all_2 = False
                        # 旋转之后如果只剩下了绝缘子，这里要经过一个判断
                        if len(valid_boxes) > 0:
                            valid_labels = [xx[4] for xx in valid_boxes]
                            if np.all(np.array(valid_labels) == 2):
                                print('valid_boxes', all_boxes_sub[valid_indices])
                                # import pdb
                                # pdb.set_trace()
                                remain_boxes = []
                                for valid_box in np.array(valid_boxes).astype(np.int32):
                                    xmin1_new, ymin1_new, xmax1_new, ymax1_new, label = valid_box
                                    tmp_im = im_sub[ymin1_new:ymax1_new, xmin1_new:xmax1_new]
                                    minval = tmp_im.min()
                                    maxval = tmp_im.max()
                                    if maxval > minval and (
                                            np.sum(tmp_im[:, :, 0] > 0) > np.prod(tmp_im.shape[:2]) / 2):
                                        remain_boxes.append(valid_box)
                                if len(remain_boxes) == len(valid_boxes):  # 必须保证所有的绝缘子都符合前面的条件
                                    remain_boxes = np.array(remain_boxes)
                                    xmin1 = np.min(remain_boxes[:, 0])
                                    ymin1 = np.min(remain_boxes[:, 1])
                                    xmax1 = np.max(remain_boxes[:, 2])
                                    ymax1 = np.max(remain_boxes[:, 3])
                                    xmin1_new = max(xmin1 - 5, 0)
                                    ymin1_new = max(ymin1 - 5, 0)
                                    xmax1_new = min(xmax1 + 5, width_sub - 1)
                                    ymax1_new = min(ymax1 + 5, height_sub - 1)
                                    valid_boxes = remain_boxes.tolist() + \
                                                  [[xmin1_new, ymin1_new, xmax1_new, ymax1_new, 1]]
                                    has_all_2 = True
                                else:
                                    valid_boxes = []

                        if len(valid_boxes) > 0:

                            save_prefix = '%d_%d_%d_%d' % (i, oi, augi, bi)

                            if np.random.rand() < 0.2:
                                im_sub1 = im_sub.copy()
                                for obj in valid_boxes:
                                    xmin1, ymin1, xmax1, ymax1, label = obj
                                    cv2.rectangle(im_sub1, (int(xmin1), int(ymin1)), (int(xmax1), int(ymax1)),
                                                  color=(0, 0, 255) if label == 1 else (0, 255, 0), thickness=2,
                                                  lineType=2)
                                    cv2.circle(im_sub1, center=(int(xmin1), int(ymin1)),
                                               radius=3, color=(0, 255, 0), thickness=2)
                                cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', im_sub1)

                            IsValid = False
                            for valid_box in valid_boxes:
                                xmin1_new, ymin1_new, xmax1_new, ymax1_new, label = valid_box
                                xc1 = (xmin1_new + xmax1_new) / 2.
                                yc1 = (ymin1_new + ymax1_new) / 2.
                                w1 = xmax1_new - xmin1_new
                                h1 = ymax1_new - ymin1_new

                                if w1 > 10 and h1 > 10:
                                    IsValid = True

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

                            if IsValid:
                                # for coco format
                                single_image = {}
                                single_image['file_name'] = save_prefix + '.jpg'
                                single_image['id'] = image_id
                                single_image['width'] = width_sub
                                single_image['height'] = height_sub
                                data_dict['images'].append(single_image)
                                image_id = image_id + 1

                                # for yolo format
                                cv2.imwrite(save_img_path + save_prefix + '.jpg', im_sub[:,:,::-1]) # RGB --> BGR
                                with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                                    fp.writelines(valid_lines)

                                list_lines.append('./images/%s.jpg\n' % save_prefix)
                            else:
                                continue  # 如果这一步没通过，则不进行旋转增强了

                            if has_all_2:
                                print('check this image: ', save_img_shown_path + save_prefix + '.jpg')
                                all2_image_path.append(save_img_shown_path + save_prefix + '.jpg\n')
                                # import pdb
                                # pdb.set_trace()

                            if do_aug and subset == 'train':
                                # extract rotated image patches
                                # 只有前一步条件满足，才会进行旋转增强
                                cnt = np.array([
                                    [[xmin, ymin]],
                                    [[xmax, ymin]],
                                    [[xmax, ymax]],
                                    [[xmin, ymax]]
                                ])

                                quads = rotate_rect(cnt.reshape([-1, 8]), angle_step=90)

                                # np.random.shuffle(quads)

                                for qi, quad in enumerate(quads):

                                    quad = quad.reshape([8])

                                    # print('quad2', quad)

                                    rect = cv2.minAreaRect(quad.astype(np.int32).reshape([4, 2]))
                                    # print("rect: {}".format(rect))

                                    # the order of the box points: bottom left, top left, top right,
                                    # bottom right
                                    rbox = cv2.boxPoints(rect)
                                    rbox = np.int0(rbox)
                                    rwidth = int(rect[1][0])
                                    rheight = int(rect[1][1])

                                    src_pts = quad.reshape([4, 2]).astype("float32")
                                    # print('src_pts', src_pts)
                                    # coordinate of the points in box points after the rectangle has been
                                    # straightened
                                    # dst_pts = np.array([[0, rheight-1],
                                    #                     [0, 0],
                                    #                     [rwidth-1, 0],
                                    #                     [rwidth-1, rheight-1]], dtype="float32")
                                    dst_pts = np.array([[0, 0],
                                                        [rwidth - 1, 0],
                                                        [rwidth - 1, rheight - 1],
                                                        [0, rheight - 1]
                                                        ], dtype="float32")
                                    # print('dst_pts', dst_pts)

                                    # the perspective transformation matrix
                                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                                    # print('M')
                                    # print(M)

                                    # directly warp the rotated rectangle to get the straightened rectangle
                                    warped = cv2.warpPerspective(img0, M, (rwidth, rheight))

                                    im_sub = np.zeros((1024, 1024, 3), dtype=np.uint8)
                                    im_sub[:min(1024, rheight), :min(1024, rwidth), :] = \
                                        warped.copy()[:min(1024, rheight), :min(1024, rwidth), :]

                                    imgpoly = shgeo.Polygon([(quad[0], quad[1]), (quad[2], quad[3]),
                                                             (quad[4], quad[5]), (quad[6], quad[7])])

                                    height_sub, width_sub = im_sub.shape[:2]
                                    valid_boxes = []
                                    valid_lines = []
                                    valid_indices = []
                                    for vi, obj in enumerate(all_boxes_sub):
                                        xmin1, ymin1, xmax1, ymax1, label = obj.copy()
                                        gtpoly = shgeo.Polygon([(xmin1, ymin1),
                                                                (xmax1, ymin1),
                                                                (xmax1, ymax1),
                                                                (xmin1, ymax1)])

                                        inter_poly, half_iou = calchalf_iou(gtpoly, imgpoly)

                                        if half_iou > 0.5:
                                            gt_poly = np.array([xmin1, ymin1, xmax1, ymin1, xmax1, ymax1, xmin1, ymax1],
                                                               dtype=np.float32)
                                            gt_poly = gt_poly.reshape([4, 2])
                                            gt_poly = np.concatenate([gt_poly, np.array([1, 1, 1, 1]).reshape([4, 1])],
                                                                     axis=1).T  # 3x4
                                            gt_poly_new = M @ gt_poly  # 3x4
                                            gt_poly_new = gt_poly_new.T[:, :2].reshape([4, 2])
                                            xmin1 = np.min(gt_poly_new[:, 0])
                                            ymin1 = np.min(gt_poly_new[:, 1])
                                            xmax1 = np.max(gt_poly_new[:, 0])
                                            ymax1 = np.max(gt_poly_new[:, 1])

                                            xmin1_new = max(xmin1, 0)
                                            ymin1_new = max(ymin1, 0)
                                            xmax1_new = min(xmax1, width_sub - 1)
                                            ymax1_new = min(ymax1, height_sub - 1)
                                            valid_boxes.append([xmin1_new, ymin1_new, xmax1_new, ymax1_new, label])
                                            valid_indices.append(vi)

                                    has_all_2 = False
                                    # 旋转之后如果只剩下了绝缘子，这里要经过一个判断
                                    if len(valid_boxes) > 0:
                                        valid_labels = [xx[4] for xx in valid_boxes]
                                        if np.all(np.array(valid_labels) == 2):
                                            print('valid_boxes', all_boxes_sub[valid_indices])
                                            # import pdb
                                            # pdb.set_trace()
                                            remain_boxes = []
                                            for valid_box in np.array(valid_boxes).astype(np.int32):
                                                xmin1_new, ymin1_new, xmax1_new, ymax1_new, label = valid_box
                                                tmp_im = im_sub[ymin1_new:ymax1_new, xmin1_new:xmax1_new]
                                                minval = tmp_im.min()
                                                maxval = tmp_im.max()
                                                if maxval > minval and (
                                                        np.sum(tmp_im[:, :, 0] > 0) > np.prod(tmp_im.shape[:2]) / 2):
                                                    remain_boxes.append(valid_box)
                                            if len(remain_boxes) == len(valid_boxes):  # 必须保证所有的绝缘子都符合前面的条件
                                                remain_boxes = np.array(remain_boxes)
                                                xmin1 = np.min(remain_boxes[:, 0])
                                                ymin1 = np.min(remain_boxes[:, 1])
                                                xmax1 = np.max(remain_boxes[:, 2])
                                                ymax1 = np.max(remain_boxes[:, 3])
                                                xmin1_new = max(xmin1 - 5, 0)
                                                ymin1_new = max(ymin1 - 5, 0)
                                                xmax1_new = min(xmax1 + 5, width_sub - 1)
                                                ymax1_new = min(ymax1 + 5, height_sub - 1)
                                                valid_boxes = remain_boxes.tolist() + \
                                                              [[xmin1_new, ymin1_new, xmax1_new, ymax1_new, 1]]
                                                has_all_2 = True
                                            else:
                                                valid_boxes = []

                                    if len(valid_boxes) > 0:

                                        save_prefix = '%d_%d_%d_%d_%d' % (i, oi, augi, bi, qi)

                                        if np.random.rand() < 0.2:
                                            im_sub1 = im_sub.copy()
                                            for obj in valid_boxes:
                                                xmin1, ymin1, xmax1, ymax1, label = obj
                                                cv2.rectangle(im_sub1, (int(xmin1), int(ymin1)), (int(xmax1), int(ymax1)),
                                                              color=(0, 0, 255) if label == 1 else (0, 255, 0), thickness=2,
                                                              lineType=2)
                                                cv2.circle(im_sub1, center=(int(xmin1), int(ymin1)),
                                                           radius=3, color=(0, 255, 0), thickness=2)
                                            cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', im_sub1)

                                        IsValid = False
                                        for valid_box in valid_boxes:
                                            xmin1_new, ymin1_new, xmax1_new, ymax1_new, label = valid_box
                                            xc1 = (xmin1_new + xmax1_new) / 2.
                                            yc1 = (ymin1_new + ymax1_new) / 2.
                                            w1 = xmax1_new - xmin1_new
                                            h1 = ymax1_new - ymin1_new
                                            if w1 > 10 and h1 > 10:
                                                IsValid = True

                                                valid_lines.append(
                                                    "%d %f %f %f %f\n" % (
                                                        label - 1, xc1 / 1024, yc1 / 1024, w1 / 1024, h1 / 1024))

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

                                        if IsValid:
                                            # for coco format
                                            single_image = {}
                                            single_image['file_name'] = save_prefix + '.jpg'
                                            single_image['id'] = image_id
                                            single_image['width'] = width_sub
                                            single_image['height'] = height_sub
                                            data_dict['images'].append(single_image)
                                            image_id = image_id + 1
                                            # for yolo format
                                            cv2.imwrite(save_img_path + save_prefix + '.jpg', im_sub[:, :, ::-1])  # RGB-->BGR
                                            with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                                                fp.writelines(valid_lines)

                                            list_lines.append('./images/%s.jpg\n' % save_prefix)
                                    if has_all_2:
                                        print('check this image: ', save_img_shown_path + save_prefix + '.jpg')
                                        all2_image_path.append(save_img_shown_path + save_prefix + '.jpg\n')
                                        # import pdb
                                        # pdb.set_trace()

    if len(list_lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_root + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)

    if len(all2_image_path) > 0:
        with open(save_root + '/%s_all2_images.txt' % subset, 'w') as fp:
            fp.writelines(all2_image_path)


def main1(subset='train'):
    subsize = 5120
    gap = 128
    if subset == 'train':
        orig_img_path = '/media/ubuntu/Temp/gd/data/aerial/%d_%d/' % (subsize, gap)
    elif subset == 'val':
        orig_img_path = '/media/ubuntu/Temp/gd/data/aerial2/%d_%d/' % (subsize, gap)
    anno_path = '/home/ubuntu/Downloads/Annotations/%s/' % subset
    save_root = '/media/ubuntu/Data/gd_1024/%s/' % subset
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
        print(filename, '=' * 50)

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

                    save_prefix = '%s_%d' % (filename.replace('.tif', ''), bi)

                    if True:
                        im_sub1 = im_sub.copy()
                        for obj in valid_boxes:
                            xmin1, ymin1, xmax1, ymax1, label = obj
                            cv2.rectangle(im_sub1, (int(xmin1), int(ymin1)), (int(xmax1), int(ymax1)),
                                          color=(0, 0, 255) if label == 1 else (0, 255, 0), thickness=2,
                                          lineType=2)
                        cv2.imwrite(save_img_shown_path + filename.replace('.tif', '_shown_%d.jpg' % bi), im_sub1)

                    # for coco format
                    single_image = {}
                    single_image['file_name'] = save_prefix + '.jpg'
                    single_image['id'] = image_id
                    single_image['width'] = width_sub
                    single_image['height'] = height_sub
                    data_dict['images'].append(single_image)
                    image_id = image_id + 1

                    # for yolo format
                    cv2.imwrite(save_img_path + save_prefix + '.jpg', im_sub)
                    with open(save_txt_path + save_prefix + '.txt', 'w') as fp:
                        fp.writelines(valid_lines)

                    list_lines.append('./images/%s.jpg\n' % save_prefix)

    if len(list_lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_root + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


if __name__ == '__main__':
    # main_test_gt()
    # main_test_rotate()
    main(subset='train', do_aug=True)
    main(subset='val', do_aug=False)
