import sys, os, glob

import cv2
import numpy as np
import torch
from osgeo import gdal, osr
from natsort import natsorted
from yoloV5.myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, box_iou_np, \
    box_intersection_np, load_gt_polys_from_esri_xml, compute_offsets, alpha_map
import json
import socket
from PIL import Image, ImageDraw, ImageFilter

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


def extract_fg_images(subset='train', save_root=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    fg_images_filename = os.path.join(save_root, subset + '_fg_images.npy')
    fg_boxes_filename = os.path.join(save_root, subset + '_fg_boxes.npy')
    if os.path.exists(fg_images_filename) and os.path.exists(fg_boxes_filename):
        return fg_images_filename, fg_boxes_filename

    gt_postfix = '_gt_5.xml'
    valid_labels_set = [1, 2, 3, 4]

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    cache_label_list = [1, 2, 3]
    cache_patches_list = []  # save the extracted gt_img_patches
    cache_boxes_list = []  # save the gt_boxes with the patch

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

                # crop the gt_boxes patch and the gt_boxes within it and save them for further augmentation
                if label in cache_label_list:
                    xc = (xmin0 + xmax0) // 2
                    yc = (ymin0 + ymax0) // 2
                    width, height = xmax0 - xmin0, ymax0 - ymin0
                    width = width * 1.1 + 32
                    height = height * 1.1 + 32
                    xoffset = max(0, xc - width // 2)
                    yoffset = max(0, yc - height // 2)
                    if xoffset + width > orig_width:
                        width = orig_width - xoffset
                    if yoffset + height > orig_height:
                        height = orig_height - yoffset
                    # find the gt_boxes within this box
                    boxes = np.copy(gt_boxes)
                    labels = np.copy(gt_labels)
                    ious = box_iou_np(np.array([xoffset, yoffset, xoffset + width, yoffset + height],
                                               dtype=np.float32).reshape(-1, 4),
                                      boxes)
                    idx2 = np.where(ious > 1e-8)[1]
                    tmp_boxes = []
                    if len(idx2) > 0:
                        valid_boxes = boxes[idx2, :]
                        valid_labels = labels[idx2]
                        valid_boxes[:, [0, 2]] -= xoffset
                        valid_boxes[:, [1, 3]] -= yoffset
                        for box1, label1 in zip(valid_boxes.astype(np.int32), valid_labels):
                            xmin, ymin, xmax, ymax = box1
                            xmin1 = max(1, xmin)
                            ymin1 = max(1, ymin)
                            xmax1 = min(xmax, width - 1)
                            ymax1 = min(ymax, height - 1)
                            # here, check the new gt_box[xmin1, ymin1, xmax1, ymax1]
                            # if the area of new gt_box is less than 0.6 of the original box, then remove this box and
                            # record its position, to put it to zero in the image
                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                            area = (xmax - xmin) * (ymax - ymin)
                            if area1 >= 0.6 * area:
                                tmp_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                    if len(tmp_boxes) > 0:
                        cutout = []
                        for bi in range(3):
                            band = ds.GetRasterBand(bi + 1)
                            band_data = band.ReadAsArray(int(xoffset), int(yoffset),
                                                         win_xsize=int(width),
                                                         win_ysize=int(height))
                            cutout.append(band_data)
                        cutout = np.stack(cutout, -1)  # this is RGB
                        cache_patches_list.append(cutout)
                        cache_boxes_list.append(np.array(tmp_boxes).reshape(-1, 5))

    if len(cache_patches_list) > 0:
        np.save(fg_images_filename, cache_patches_list, allow_pickle=True)
        np.save(fg_boxes_filename, cache_boxes_list, allow_pickle=True)
    return fg_images_filename, fg_boxes_filename


def extract_bg_images(subset='train', save_root=None, random_count=0):
    hostname = socket.gethostname()
    if hostname == 'master':
        source = '/media/ubuntu/Data/%s_list.txt' % (subset)
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    else:
        source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
        gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    save_root = '%s/bg_images/' % save_root
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    elif len(glob.glob(save_root + '/*.png')) > 0:
        return save_root

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    if random_count > 0:
        # based on the cached image patches
        # first extract the background image
        # then paste the image patch into the bg image
        big_subsize = 10240
        gt_gap = 128
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

            gt_boxes, gt_boxes_labels = load_gt(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                valid_labels=[1, 2, 3, 4])

            gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_LineRegion14.xml')

            gt_polys, gt_labels = load_gt_polys_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform,
                                                              mapcoords2pixelcoords=True)
            print(len(gt_polys), len(gt_labels))

            # 首先根据标注生成mask图像，存在内存问题！！！
            print('generate mask ...')
            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            if True:
                # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
                # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)

                if len(gt_boxes) > 0:
                    for box in gt_boxes:
                        xmin, ymin, xmax, ymax = box
                        poly = np.array([[xmin, ymin], [xmax, ymin],
                                         [xmax, ymax], [xmin, ymax]], dtype=np.int32).reshape([4, 2])
                        cv2.drawContours(mask, [poly], -1, color=(255, 0, 0), thickness=-1)

                for poly, label in zip(gt_polys, gt_labels):  # poly为nx2的点, numpy.array
                    cv2.drawContours(mask, [poly], -1, color=(255, 0, 0), thickness=-1)

                # mask_savefilename = save_root + "/" + file_prefix + ".png"
                # cv2.imwrite(mask_savefilename, mask)
                # if not os.path.exists(mask_savefilename):
                #     cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

            # 在图片中随机采样一些点
            print('generate random sample points ...')
            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
            random_indices_y = []
            random_indices_x = []
            for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up
                # sub_width = min(orig_width, big_subsize)
                # sub_height = min(orig_height, big_subsize)
                # if xoffset + sub_width > orig_width:
                #     sub_width = orig_width - xoffset
                # if yoffset + sub_height > orig_height:
                #     sub_height = orig_height - yoffset

                # print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
                img0 = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
                for b in range(3):
                    band = ds.GetRasterBand(b + 1)
                    img0[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)
                img0_sum = np.sum(img0, axis=2)
                indices_y, indices_x = np.where(img0_sum > 0)
                inds = np.arange(len(indices_x))
                np.random.shuffle(inds)
                count = min(random_count, len(inds))
                random_indices_y.append(indices_y[inds[:count]] + yoffset)
                random_indices_x.append(indices_x[inds[:count]] + xoffset)

                del img0, img0_sum, indices_y, indices_x

            random_indices_y = np.concatenate(random_indices_y).reshape(-1, 1)
            random_indices_x = np.concatenate(random_indices_x).reshape(-1, 1)
            print(random_indices_y.shape, random_indices_x.shape)
            print(random_indices_y[:10])
            print(random_indices_x[:10])

            for j, (xc, yc) in enumerate(zip(random_indices_x, random_indices_y)):  # per item

                w = np.random.randint(low=600, high=1024)
                h = np.random.randint(low=600, high=1024)
                xmin1, ymin1 = xc - w / 2, yc - h / 2
                xmax1, ymax1 = xc + w / 2, yc + h / 2
                if xmin1 < 0:
                    xmin1 = 0
                    xmax1 = w
                if ymin1 < 0:
                    ymin1 = 0
                    ymax1 = h
                if xmax1 > orig_width - 1:
                    xmax1 = orig_width - 1
                    xmin1 = orig_width - 1 - w
                if ymax1 > orig_height - 1:
                    ymax1 = orig_height - 1
                    ymin1 = orig_height - 1 - h

                xmin1 = int(xmin1)
                xmax1 = int(xmax1)
                ymin1 = int(ymin1)
                ymax1 = int(ymax1)
                width = xmax1 - xmin1
                height = ymax1 - ymin1

                # 查找gtboxes里面，与当前框有交集的框
                mask1 = mask[ymin1:ymax1, xmin1:xmax1]
                if mask1.sum() > 0:
                    continue

                cutout = []
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(xmin1, ymin1, win_xsize=width, win_ysize=height)
                    cutout.append(band_data)
                im1 = np.stack(cutout, -1)  # RGB

                cv2.imencode('.png', im1)[1].tofile('%s/bg_%d_%d.png' % (save_root, ti, j))

            del mask
    return save_root


def compose_fg_bg_v1(bg, fg_images_list, fg_boxes_list, inds):   # not good
    # bg: HxWx3 RGB
    # fg_ims: list of RGB images [hxwx3]
    # fg_boxes: list of boxes [nx5]

    im = np.copy(bg)
    H, W = im.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    boxes = []
    for ind in inds:
        fg = np.zeros((H, W, 3), dtype=np.uint8)
        fg_im = np.copy(fg_images_list[ind])  # RGB
        fg_boxes = np.copy(fg_boxes_list[ind])  # nx5
        if len(fg_boxes) == 0:
            continue
        h, w = fg_im.shape[:2]
        area = h * w
        is_ok = False
        xc, yc = -1, -1
        for step in range(10):
            mask1 = np.zeros((H, W), dtype=np.uint8)
            xc = np.random.randint(low=int(w//2+1), high=int(W-w//2-1))
            yc = np.random.randint(low=int(h//2+1), high=int(H-h//2-1))
            left = xc - w//2
            up = yc - h//2
            mask1[up:(up+h), left:(left+h)] = 1
            area1 = len(np.where(mask & mask1)[0])
            if area1 < 0.4 * area:
                is_ok = True
                break
        if is_ok:
            left = xc - w//2
            up = yc - h//2
            alpha = alpha_map(W, H, w, h, xc, yc)
            fg[up:(up+h), left:(left+w), :] = fg_im
            mask[up:(up + h), left:(left + h)] = 1
            im = alpha * fg + (1 - alpha) * im
            im = im.astype(np.uint8)

            fg_boxes[:, [0, 2]] += left
            fg_boxes[:, [1, 3]] += up
            boxes.append(fg_boxes)
    if len(boxes) > 0:
        boxes = np.concatenate(boxes, axis=0)
        return im, boxes  # RGB, nx5
    else:
        return [], []


def compose_fg_bg(bg, fg_images_list, fg_boxes_list, inds):   # not good
    # bg: HxWx3 RGB
    # fg_ims: list of RGB images [hxwx3]
    # fg_boxes: list of boxes [nx5]

    im = Image.fromarray(np.copy(bg))
    W, H = im.size
    mask = np.zeros((H, W), dtype=np.uint8)
    boxes = []
    blur_radius = 10
    for ind in inds:
        fg_im = np.copy(fg_images_list[ind])
        h, w = fg_im.shape[:2]
        if h > H - 32 or w > W - 32:
            continue

        fg_boxes0 = np.copy(fg_boxes_list[ind])  # nx5
        valid_inds = np.where(fg_boxes0[:, 4] >= 2)[0]
        fg_boxes = fg_boxes0[valid_inds, :5]

        if len(fg_boxes) == 0:
            continue
        # print('fg_boxes', fg_boxes)

        fg_boxes_xmin = np.min(fg_boxes[:, 0])
        fg_boxes_ymin = np.min(fg_boxes[:, 1])
        fg_boxes_xmax = np.max(fg_boxes[:, 2])
        fg_boxes_ymax = np.max(fg_boxes[:, 3])

        is_ok = False
        xc, yc = -1, -1
        for step in range(10):
            xc = np.random.randint(low=int(w//2+1), high=int(W-w//2-1))
            yc = np.random.randint(low=int(h//2+1), high=int(H-h//2-1))
            left = xc - w//2
            up = yc - h//2
            if len(boxes) == 0:
                is_ok = True
                break
            mask1 = np.zeros((H, W), dtype=np.uint8)
            mask1[up:(up+h), left:(left+w)] = 1
            area1 = len(np.where(mask & mask1)[0])
            if area1 < 10:
                is_ok = True
                break
        if is_ok:
            left = xc - w//2
            up = yc - h//2

            fg = np.zeros((H, W, 3), dtype=np.uint8)
            mask1 = np.zeros((H, W), dtype=np.uint8)
            fg[up:(up+h), left:(left+w), :] = fg_im
            mask[up:(up+h), left:(left+w)] = 1
            mask1_im = Image.fromarray(mask1)
            draw1 = ImageDraw.Draw(mask1_im)
            draw1.rectangle((left+fg_boxes_xmin, up+fg_boxes_ymin, left+fg_boxes_xmax, up+fg_boxes_ymax), fill=255)
            mask1_im_blur = mask1_im.filter(ImageFilter.GaussianBlur(blur_radius))
            im.paste(Image.fromarray(fg), (0, 0), mask1_im_blur)

            fg_boxes[:, [0, 2]] += left
            fg_boxes[:, [1, 3]] += up
            boxes.append(fg_boxes)
    if len(boxes) > 0:
        boxes = np.concatenate(boxes, axis=0)
        return np.array(im), boxes  # RGB, nx5
    else:
        return [], []


def compose_fg_bg_images(subset='train', aug_times=1, save_img=False,
                         save_root=None, do_rot=False, random_count=0):
    save_root = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    save_img_path = '%s/images/' % save_root
    save_img_shown_path = '%s/images_shown/' % save_root
    save_txt_path = '%s/labels/' % save_root
    for p in [save_img_path, save_txt_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    print('extract fg images ...')
    fg_images_filename, fg_boxes_filename = extract_fg_images(subset, save_root)

    print('extract bg images ...')
    bg_images_dir = extract_bg_images(subset, save_root, random_count)

    print(fg_images_filename)
    print(fg_images_filename)
    print(bg_images_dir)
    print('compose fg and bg to new train images ...')
    fg_images_list = np.load(fg_images_filename, allow_pickle=True)  # list of RGB images [HxWx3]
    fg_boxes_list = np.load(fg_boxes_filename, allow_pickle=True)  # list of boxes [nx5]
    fg_inds = np.arange(len(fg_images_list))

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

    bg_filenames = glob.glob(bg_images_dir + '/*.png')
    bg_count = min(5000, len(bg_filenames))
    if bg_count > 5000:
        bg_indices = np.random.choice(np.arange(len(bg_filenames)), size=bg_count, replace=False)
    else:
        bg_indices = np.arange(len(bg_filenames))
    for bg_ind in bg_indices:
        bg_filename = bg_filenames[bg_ind]
        file_prefix = bg_filename.split(os.sep)[-1].replace('.png', '')
        bg = cv2.imread(bg_filename)

        if min(bg.shape[:2]) < 500:
            continue

        for aug_time in range(aug_times):
            selected_fg_inds = np.random.choice(fg_inds, size=np.random.randint(1, 4))

            im, gt_boxes = compose_fg_bg(bg, fg_images_list, fg_boxes_list, selected_fg_inds)

            save_img = False

            # draw gt boxes
            if len(gt_boxes) > 0:
                save_prefix = '%s_%d' % (file_prefix, aug_time)
                sub_h, sub_w = im.shape[:2]

                # save image
                # for coco format
                single_image = {}
                single_image['file_name'] = save_prefix + '.jpg'
                single_image['id'] = image_id
                single_image['width'] = sub_w
                single_image['height'] = sub_h
                data_dict['images'].append(single_image)

                # for yolo format
                cv2.imwrite(save_img_path + save_prefix + '.jpg', im[:, :, ::-1])  # RGB --> BGR

                list_lines.append('./images/%s.jpg\n' % save_prefix)

                if np.random.rand() < 0.05:
                    save_img = True

                valid_lines = []
                for box2 in gt_boxes.astype(np.int32):
                    xmin, ymin, xmax, ymax, label = box2

                    if save_img:
                        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color=colors[label],
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
                    cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', im[:, :, ::-1])  # RGB --> BGR

            # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
            # im /= 255.0  # 0 - 255 to 0.0 - 1.0
            # ims.append(im)

    if len(list_lines) > 0:
        with open(save_root + '/%s.txt' % subset, 'w') as fp:
            fp.writelines(list_lines)

        with open(save_root + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


def main(subset='train', aug_times=1, save_img=False,
         save_root=None, do_rot=False, random_count=0):
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

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

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
                    box_w, box_h = int(xmax - xmin), int(ymax - ymin)

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
                                    sub_gt_boxes = [box3 for ii, box3 in enumerate(sub_gt_boxes) if
                                                    ii not in idx2.tolist()]

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

    print(sys.argv)
    if len(sys.argv) != 6:
        print('python ./this_script.py aug_type aug_times(int) save_img(int) do_rot(int) random_count')
        sys.exit(-1)
    aug_type = sys.argv[1]
    aug_times = int(sys.argv[2])
    save_img = int(sys.argv[3]) != 0
    do_rot = int(sys.argv[4]) != 0
    random_count = int(sys.argv[5])

    hostname = socket.gethostname()

    if hostname == 'master':
        save_root = '/media/ubuntu/Data/gd_newAug%d_Rot%d_4classes_%s' % (aug_times, do_rot, aug_type)
    else:
        save_root = 'E:/gd_newAug%d_Rot%d_4classes_%s' % (aug_times, do_rot, aug_type)

    if aug_type == 'v1':

        # TODO zzs, implement the rotate augmentation

        main(subset='train', aug_times=aug_times, save_img=save_img, save_root=save_root,
             do_rot=do_rot, random_count=random_count)
        main(subset='val', aug_times=2, save_img=save_img, save_root=save_root,
             do_rot=do_rot, random_count=1)

    elif aug_type == 'v2':

        compose_fg_bg_images(subset='train', aug_times=aug_times, save_img=save_img, save_root=save_root,
                             do_rot=do_rot, random_count=random_count)
        compose_fg_bg_images(subset='val', aug_times=1, save_img=True, save_root=save_root,
                             do_rot=do_rot, random_count=4)
