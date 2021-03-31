
import os
from PIL import Image
import xml.dom.minidom
import numpy as np
import cv2
import shapely.geometry as shgeo
from shapely.geometry import Point
import json
from natsort import natsorted


def calchalf_iou( poly1, poly2):
    """
        It is not the iou on usual, the iou is the value of intersection over poly1
    """
    inter_poly = poly1.intersection(poly2)
    inter_area = inter_poly.area
    poly1_area = poly1.area
    half_iou = inter_area / poly1_area
    return inter_poly, half_iou


def main(subset='train'):
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
    main(subset='val')