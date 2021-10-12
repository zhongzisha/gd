import sys, os, glob, shutil
import numpy as np
import cv2
import argparse
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
import multiprocessing
from multiprocessing import Pool
import itertools
import xml.dom.minidom


def add_line(im, mask, p0s, p1s):
    line_width = np.random.randint(1, 3)
    for p0, p1 in zip(p0s, p1s):

        d = np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
        if d < 30:
            continue

        r = np.random.randint(160, 220)
        g = r - np.random.randint(1, 10)
        b = g - np.random.randint(1, 10)

        cv2.line(im, p0, p1, color=(r, g, b), thickness=line_width)
        cv2.line(mask, p0, p1, color=(1, 0, 0), thickness=line_width)

    return im, mask


def paste_fg_images_to_bg(im, mask, fg_images, count=10):
    im = im.copy()
    mask = mask.copy()
    boxes = []

    H, W = im.shape[:2]
    boxes_mask = np.zeros((H, W), dtype=np.uint8)
    im_pil = Image.fromarray(im)
    fg_all_pil = Image.new('RGBA', im_pil.size)
    for step in range(count):
        Ys, Xs = np.where(mask)
        indices = np.arange(len(Ys))
        if len(Ys) == 0:
            continue

        fg = np.random.choice(fg_images)
        bgr = fg[:, :, :3]
        alpha = fg[:, :, 3]
        w, h = np.random.randint(20, 50), np.random.randint(20, 50)
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)

        # 改成灰灰的白白的
        r = np.random.randint(low=190, high=230)
        g = r - np.random.randint(1, 10)
        b = g - np.random.randint(1, 10)
        bgr[:, :, 0] = b
        bgr[:, :, 1] = g
        bgr[:, :, 2] = r

        ksize = np.random.choice([3, 5, 7, 9])
        sigmaX = np.random.choice([0.1, 1, 10])
        bgr = cv2.GaussianBlur(bgr, ksize=(ksize, ksize), sigmaX=sigmaX)

        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)
        print(bgr.shape, alpha.shape)

        prob = np.random.rand()
        if prob < 0.2:
            bgr = bgr[:, :, np.random.permutation([0, 1, 2])]
        if 0.2 < prob < 0.5:
            bgr = bgr[:, :, np.random.choice([0, 1, 2], size=3)]

        fg = np.concatenate([bgr, alpha[:, :, None]], axis=-1)
        fg_pil = Image.fromarray(fg).convert('RGBA')
        kkkk = 0
        cx, cy = 0, 0
        while True:
            if kkkk > 10:
                break
            index = np.random.choice(indices)
            cy, cx = Ys[index], Xs[index]
            if cy < 30 or cx < 30 or H - cy < 30 or W - cx < 30:
                kkkk += 1
                cx, cy = 0, 0
                continue
            else:
                break
        if cx > 0 and cy > 0:
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            im_pil.paste(fg_pil, (x1, y1), fg_pil)
            fg_all_pil.paste(fg_pil, (x1, y1), fg_pil)
            boxes_mask[y1:y2, x1:x2] = alpha > 0
            mask[y1:y2, x1:x2] = 0
            boxes.append([x1, y1, x2, y2])

    im = np.array(im_pil)
    new_img = Image.alpha_composite(im_pil.convert('RGBA'), fg_all_pil)
    im_com = np.array(new_img)

    fg_all_pil = np.array(fg_all_pil)
    alpha = fg_all_pil[:, :, 3].astype(np.float32) / 255.
    alpha = np.stack([alpha, alpha, alpha], axis=-1)
    fg_all_pil_rgb = fg_all_pil[:, :, :3]
    new_img = im * (1 - alpha) + fg_all_pil_rgb * alpha
    im_blend = new_img.astype(np.uint8)

    return im, im_com, im_blend, mask, boxes, boxes_mask


def add_line_to_image(im, crop_width=0, crop_height=0, subset='train',
                      foreign_fg_filenames=None):
    # extract sub image from im and add line to the image
    # return the croppped image and its line mask
    H, W = im.shape[:2]
    # im_sub = im[(H//2-crop_height//2):(H//2-crop_height//2+crop_height),
    #          (W//2-crop_width//2):(W//2-crop_width//2+crop_width), :]

    if crop_width != 0 or crop_height != 0:
        xc = np.random.randint(low=(crop_width + 1) // 2, high=(W - (crop_width + 1) // 2 - 1))
        yc = np.random.randint(low=(crop_height + 1) // 2, high=(H - (crop_height + 1) // 2 - 1))
        im_sub = im[(yc - crop_height // 2):(yc - crop_height // 2 + crop_height),
                 (xc - crop_width // 2):(xc - crop_width // 2 + crop_width),
                 :]
        if len(np.unique(im_sub)) < 50:
            return None, None, None, None
        H, W = crop_height, crop_width
    else:
        im_sub = im

    mask = np.zeros((H, W), dtype=np.uint8)
    total_line_count = np.random.randint(2, 5) if 'train' in subset else 1
    for step in range(total_line_count):
        y = np.random.randint(0, H - 1, size=2)
        x = np.random.randint(0, W - 1, size=2)

        x1, x2 = x
        y1, y2 = y
        if abs(x1 - x2) == 1 and (20 < x1 < W - 20):
            expand = np.random.randint(low=10, high=x1 - 9, size=2)
            x1_l = x1 - expand[0]
            x1_r = x1 + expand[1]
            p0s, p1s = [], []
            if x1_l >= 3:
                p0s.append((x1_l, 0))
                p1s.append((x1_l, H - 1))
            p0s.append((x1, 0))
            p1s.append((x1, H - 1))
            if x1_r <= W - 3:
                p0s.append((x1_r, 0))
                p1s.append((x1_r, H - 1))

            im_sub, mask = add_line(im_sub, mask, p0s, p1s)
        elif abs(y1 - y2) == 1 and (20 < y1 < H - 20):
            expand = np.random.randint(low=10, high=y1 - 9, size=2)
            y1_u = y1 - expand[0]
            y1_b = y1 + expand[1]
            p0s, p1s = [], []
            if y1_u >= 3:
                p0s.append((0, y1_u))
                p1s.append((W - 1, y1_u))
            p0s.append((0, y1))
            p1s.append((W - 1, y1))
            if y1_b <= H - 3:
                p0s.append((0, y1_b))
                p1s.append((W - 1, y1_b))

            im_sub, mask = add_line(im_sub, mask, p0s, p1s)
        elif abs(x1 - x2) > 10 and abs(y1 - y2) > 10:
            k = (y1 - y2) / (x2 - x1)
            b = - k * x1 - y1
            # (y1 - y2)x + (x1 - x2)y + x1y2 -y1x2 = 0
            expand = np.random.randint(low=10, high=100, size=2)
            b_up = b - expand[0]
            b_down = b + expand[1]
            if abs(k) > 1:
                # y=3, y=H-3
                p0s = [(int((-3 - bb) / k), 3) for bb in [b, b_up, b_down]]
                p1s = [(int((-H + 3 - bb) / k), H - 3) for bb in [b, b_up, b_down]]
                im_sub, mask = add_line(im_sub, mask, p0s, p1s)
            elif 1 >= abs(k) > 0.05:
                # x=3, x=W-3
                p0s = [(3, int(-k * 3 - bb)) for bb in [b, b_up, b_down]]
                p1s = [(W - 3, int(-k * (W - 3) - bb)) for bb in [b, b_up, b_down]]
                im_sub, mask = add_line(im_sub, mask, p0s, p1s)

    # blur the image
    prob = np.random.rand()
    if prob < 0.5:
        ksize = np.random.choice([3, 5, 7, 9])
        sigmas = np.arange(0.5, ksize, step=0.5)
        im_sub = cv2.GaussianBlur(im_sub, ksize=(ksize, ksize),
                                  sigmaX=np.random.choice(sigmas),
                                  sigmaY=np.random.choice(sigmas))
    elif 0.5 <= prob <= 0.8:
        #     ksize = np.random.choice([3, 5])
        #     im_sub = cv2.medianBlur(im_sub, ksize=ksize)
        # else:
        im_sub_with_mask = np.concatenate([im_sub, mask[:, :, None]], axis=2)
        im_sub_with_mask = elastic_transform_v2(im_sub_with_mask, im_sub.shape[1] * 2,
                                                im_sub.shape[1] * np.random.randint(low=4, high=8) / 100,
                                                im_sub.shape[1] * np.random.randint(low=4, high=8) / 100)
        im_sub, mask = im_sub_with_mask[:, :, :3], im_sub_with_mask[:, :, 3]

    if mask.sum() < min(mask.shape[:2]) / 2:
        return None, None, None, None

    # add foreigne objects to images
    if foreign_fg_filenames is not None:

        fg_images = []
        total_count = np.random.randint(5, 20) if 'train' in subset else 5
        fg_count = np.random.randint(low=5, high=10) if 'train' in subset else np.random.randint(low=2, high=7)
        while True:
            random_filenames = np.random.choice(foreign_fg_filenames,
                                                size=total_count,
                                                replace=True)
            for name in random_filenames:
                if len(fg_images) > total_count:
                    break
                fg_im = cv2.imread(name, cv2.IMREAD_UNCHANGED)  # BGRA
                if fg_im.shape[2] != 4:
                    continue
                x, y = np.where(fg_im[:, :, -1])
                xmin, ymin, xmax, ymax = np.min(x), np.min(y), np.max(x), np.max(y)
                if xmax - xmin < 10 or ymax - ymin < 10:
                    continue
                fg_images.append(fg_im[xmin:xmax, ymin:ymax, :])

            if len(fg_images) > 1:
                break
        print('num_fg_images: ', len(fg_images))
        # paste the fg_image to im_sub
        im_sub1, im_sub1_com, im_sub1_blend, mask1, boxes1, boxes1_mask = \
            paste_fg_images_to_bg(im_sub, mask, fg_images, fg_count)

        mask2 = mask.copy()
        mask2[np.where(boxes1_mask)] = 2

        return im_sub1, mask1, mask2, boxes1

    return im_sub, mask, None, None


def refine_line_aug(subset='train', aug_times=1, save_root=None,
                    crop_height=512, crop_width=512,
                    fg_images_filename=None, fg_boxes_filename=None,
                    bg_images_dir=None, random_count=1000,
                    foreign_fg_dir=None):
    save_dir = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = '%s/images/' % (save_dir)
    images_shown_root = '%s/images_shown/' % (save_dir)
    labels_root = '%s/annotations/' % (save_dir)
    labels_with_foreign_root = '%s/annotations_with_foreign/' % (save_dir)
    for p in [images_root, labels_root, images_shown_root, labels_with_foreign_root]:
        if not os.path.exists(p):
            os.makedirs(p)

    aug_times = aug_times if 'train' in subset else 1
    bg_filenames = glob.glob(bg_images_dir + '/*.png')
    foreign_fg_filenames = glob.glob(foreign_fg_dir + '/*.png')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["1"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    palette = np.array([[0,0,0],[255,255,255], [255,255,0],[0,255,255]])

    lines = []
    for aug_time in range(aug_times):
        if len(bg_filenames) > random_count:
            bg_indices = np.random.choice(np.arange(len(bg_filenames)), size=random_count, replace=False)
        else:
            bg_indices = np.arange(len(bg_filenames))

        for bg_ind in bg_indices:
            bg_filename = bg_filenames[bg_ind]
            file_prefix = bg_filename.split(os.sep)[-1].replace('.png', '')
            print(bg_ind, file_prefix)
            bg = cv2.imread(bg_filename)

            if min(bg.shape[:2]) < 512:
                continue

            im1, mask1, mask2, boxes1 = add_line_to_image(bg, crop_height, crop_width, subset=subset,
                                                          foreign_fg_filenames=foreign_fg_filenames)

            if im1 is None:
                continue
            if mask1.sum() < min(im1.shape[:2]) / 2:
                continue

            save_prefix = '%s_%d_%010d' % (file_prefix, aug_time, bg_ind)
            cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1[:, :, ::-1])  # 不能有中文
            cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)
            cv2.imwrite('%s/%s.png' % (labels_with_foreign_root, save_prefix), mask2)

            lines.append('%s\n' % save_prefix)

            if len(boxes1) > 0:
                sub_h, sub_w = im1.shape[:2]
                # save image
                # for coco format
                single_image = {}
                single_image['file_name'] = save_prefix + '.jpg'
                single_image['id'] = image_id
                single_image['width'] = sub_w
                single_image['height'] = sub_h
                data_dict['images'].append(single_image)

                mask2_color = np.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    mask2_color[mask2 == label, :] = color

                im2 = im1.copy()
                for box2 in boxes1:
                    xmin, ymin, xmax, ymax = box2

                    if True:
                        cv2.rectangle(im2, (xmin, ymin), (xmax, ymax), color=(255, 255, 255),
                                      thickness=2)

                    xc1 = int((xmin + xmax) / 2)
                    yc1 = int((ymin + ymax) / 2)
                    w1 = xmax - xmin
                    h1 = ymax - ymin
                    # for coco format
                    single_obj = {'area': int(w1 * h1),
                                  'category_id': 1,
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

                if True:  # np.random.rand() < 0.01:
                    cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                                np.concatenate([im2[:, :, ::-1],
                                                255 * np.stack([mask1, mask1, mask1], axis=2),
                                                mask2_color],
                                               axis=1))  # 不能有中文

                image_id = image_id + 1

    if len(lines) > 0:
        with open('%s/%s.txt' % (save_root, subset), 'w') as fp:
            fp.writelines(lines)

        with open(save_dir + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


def process_one(bg_ind, bg_filenames, subset, foreign_fg_filenames, aug_time, images_root, labels_root,
                labels_with_foreign_root, images_shown_root):
    bg_filename = bg_filenames[bg_ind]
    file_prefix = bg_filename.split(os.sep)[-1].replace('.png', '')
    bg = cv2.imread(bg_filename)
    palette = np.array([[0,0,0],[255,255,255], [255,255,0],[0,255,255]])

    if min(bg.shape[:2]) < 512:
        return None, None, None

    im1, mask1, mask2, boxes1 = add_line_to_image(bg, crop_height=0, crop_width=0, subset=subset,
                                                  foreign_fg_filenames=foreign_fg_filenames)

    if im1 is None:
        return None, None, None
    if mask1.sum() < min(im1.shape[:2]) / 2:
        return None, None, None

    save_prefix = '%s_%d_%010d' % (file_prefix, aug_time, bg_ind)
    cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1[:, :, ::-1])  # 不能有中文
    cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)
    cv2.imwrite('%s/%s.png' % (labels_with_foreign_root, save_prefix), mask2)

    mask2_color = np.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        mask2_color[mask2 == label, :] = color

    for box2 in boxes1:
        xmin, ymin, xmax, ymax = box2
        cv2.rectangle(im1, (xmin, ymin), (xmax, ymax), color=(255, 255, 255), thickness=2)
    cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                np.concatenate([im1[:, :, ::-1],
                                255 * np.stack([mask1, mask1, mask1], axis=2),
                                mask2_color], axis=1))  # 不能有中文

    return save_prefix, im1.shape[:2], boxes1


def refine_line_aug_parallel(subset='train', aug_times=1, save_root=None,
                             crop_height=512, crop_width=512,
                             fg_images_filename=None, fg_boxes_filename=None,
                             bg_images_dir=None, random_count=1000,
                             foreign_fg_dir=None):
    save_dir = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = '%s/images/' % (save_dir)
    images_shown_root = '%s/images_shown/' % (save_dir)
    labels_root = '%s/annotations/' % (save_dir)
    labels_with_foreign_root = '%s/annotations_with_foreign/' % (save_dir)
    for p in [images_root, labels_root, images_shown_root, labels_with_foreign_root]:
        if not os.path.exists(p):
            os.makedirs(p)

    aug_times = aug_times if 'train' in subset else 1
    bg_filenames = glob.glob(bg_images_dir + '/*.png')
    foreign_fg_filenames = glob.glob(foreign_fg_dir + '/*.png')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["1"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    lines = []
    for aug_time in range(aug_times):
        if len(bg_filenames) > random_count:
            bg_indices = np.random.choice(np.arange(len(bg_filenames)), size=random_count, replace=False)
        else:
            bg_indices = np.arange(len(bg_filenames))

        all_params = zip(
            bg_indices,
            itertools.repeat(bg_filenames),
            itertools.repeat(subset),
            itertools.repeat(foreign_fg_filenames),
            itertools.repeat(aug_time),
            itertools.repeat(images_root),
            itertools.repeat(labels_root),
            itertools.repeat(labels_with_foreign_root),
            itertools.repeat(images_shown_root)
        )

        with Pool(2) as p:
            results = p.starmap(process_one, all_params)

        for _, x in enumerate(results):
            save_prefix, shape, boxes1 = x
            if save_prefix is None:
                continue
            lines.append('%s\n' % save_prefix)

            if len(boxes1) > 0:
                sub_h, sub_w = shape
                # save image
                # for coco format
                single_image = {}
                single_image['file_name'] = save_prefix + '.jpg'
                single_image['id'] = image_id
                single_image['width'] = sub_w
                single_image['height'] = sub_h
                data_dict['images'].append(single_image)

                for box2 in boxes1:
                    xmin, ymin, xmax, ymax = box2

                    xc1 = int((xmin + xmax) / 2)
                    yc1 = int((ymin + ymax) / 2)
                    w1 = xmax - xmin
                    h1 = ymax - ymin
                    # for coco format
                    single_obj = {'area': int(w1 * h1),
                                  'category_id': 1,
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

    if len(lines) > 0:
        with open('%s/%s.txt' % (save_root, subset), 'w') as fp:
            fp.writelines(lines)

        with open(save_dir + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)


def test_paste():
    save_dir = '/media/ubuntu/Data/line_foreign_object_detection/check_paster'
    im_dir = '/media/ubuntu/Data/line_foreign_object_detection/augmented_data/train/images'
    mask_dir = '/media/ubuntu/Data/line_foreign_object_detection/augmented_data/train/annotations'
    im_filenames = glob.glob(os.path.join(im_dir, '*.jpg'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for step in range(10):
        filename = np.random.choice(im_filenames)
        file_prefix = filename.split(os.sep)[-1].replace('.jpg', '')
        im_sub = cv2.imread(filename)
        mask = cv2.imread(os.path.join(mask_dir, file_prefix + '.png'))[:, :, 0]

        # add foreigne objects to images
        foreign_fg_dir = '/media/ubuntu/Data/line_foreign_object_detection/fg_images'
        foreign_fg_filenames = glob.glob(foreign_fg_dir + '/*.png')
        fg_images = []
        while True:
            random_filenames = np.random.choice(foreign_fg_filenames,
                                                size=np.random.randint(2, 10),
                                                replace=True)
            for name in random_filenames:
                if len(fg_images) > 10:
                    break
                fg_im = cv2.imread(name, cv2.IMREAD_UNCHANGED)  # BGRA
                if fg_im.shape[2] != 4:
                    continue
                x, y = np.where(fg_im[:, :, -1])
                xmin, ymin, xmax, ymax = np.min(x), np.min(y), np.max(x), np.max(y)
                if xmax - xmin < 10 or ymax - ymin < 10:
                    continue
                fg_images.append(fg_im[xmin:xmax, ymin:ymax, :])

            if len(fg_images) > 1:
                break
        print('num_fg_images: ', len(fg_images))
        # paste the fg_image to im_sub
        im_sub1, im_sub1_com, im_sub1_blend, mask1, boxes1, boxes1_mask = paste_fg_images_to_bg(im_sub, mask, fg_images)

        # import pdb
        # pdb.set_trace()
        cv2.imwrite(os.path.join(save_dir, '%s_%d.png' % (file_prefix, step)), im_sub1)
        cv2.imwrite(os.path.join(save_dir, '%s_%d_com.png' % (file_prefix, step)), im_sub1_com)
        cv2.imwrite(os.path.join(save_dir, '%s_%d_blend.png' % (file_prefix, step)), im_sub1_blend)
        cv2.imwrite(os.path.join(save_dir, '%s_%d_LineMask.png' % (file_prefix, step)), mask1 * 255)
        cv2.imwrite(os.path.join(save_dir, '%s_%d_boxes_mask.png' % (file_prefix, step)), boxes1_mask * 255)
        for box in boxes1:
            x1, y1, x2, y2 = box
            cv2.rectangle(im_sub1, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)
        cv2.imwrite(os.path.join(save_dir, '%s_%d_with_boxes.png' % (file_prefix, step)), im_sub1)


def generate_test_images():
    tiffile = 'G:\\gddata\\all\\2-WV03-在建杆塔.tif'
    gt_dir = 'G:\\gddata\\all'
    save_dir = 'E:\\generated_big_test_images'
    foreign_fg_dir = 'E:/line_foreign_object_detection/fg_images'
    foreign_fg_filenames = glob.glob(foreign_fg_dir + '/*.png')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

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
    mapcoords2pixelcoords = True
    print('loading gt ...')
    gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_linepoint.xml')

    # gt_polys, gt_labels = load_gt_polys_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform,
    #                                                   mapcoords2pixelcoords=mapcoords2pixelcoords)
    DomTree = xml.dom.minidom.parse(gt_xml_filename)
    annotation = DomTree.documentElement
    regionlist = annotation.getElementsByTagName('Region')
    gt_polys = []
    gt_labels = []

    for region in regionlist:
        name = region.getAttribute("name")
        polylist = region.getElementsByTagName('Coordinates')
        for poly in polylist:
            coords_str = poly.childNodes[0].data
            coords = [float(val) for val in coords_str.strip().split(' ')]
            points = np.array(coords).reshape([-1, 2])  # nx2
            if mapcoords2pixelcoords:  # geo coordinates to pixel coordinates
                points[:, 0] -= xOrigin
                points[:, 1] -= yOrigin
                points[:, 0] /= pixelWidth
                points[:, 1] /= pixelHeight
                points += 0.5
                points = points.astype(np.int32)

            gt_polys.append(points)

    print(gt_polys)
    # import pdb
    # pdb.set_trace()
    print('load image ...')
    im_sub = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
    for b in range(3):
        band = ds.GetRasterBand(b + 1)
        im_sub[:, :, b] += band.ReadAsArray(0, 0, win_xsize=orig_width, win_ysize=orig_height)

    # 首先根据标注生成mask图像，存在内存问题！！！
    print('generate mask ...')
    mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
    # if True:
    #     # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
    #     # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)
    #
    #     for poly in gt_polys:  # poly为nx2的点, numpy.array
    #         cv2.drawContours(mask, [poly], -1, color=(1, 1, 1), thickness=-1)
    #
    #     mask_savefilename = save_dir + "/" + file_prefix + ".png"
    #     # cv2.imwrite(mask_savefilename, mask)
    #     if not os.path.exists(mask_savefilename):
    #         cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

    W, H = orig_width, orig_height
    time.sleep(3)
    points = gt_polys[0]
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        print(i, x1, y1, x2, y2)
        if abs(x1 - x2) == 1 and (20 < x1 < W - 20):
            expand = np.random.randint(low=10, high=x1 - 9, size=2)
            x1_l = x1 - expand[0]
            x1_r = x1 + expand[1]
            p0s, p1s = [], []
            if x1_l >= 3:
                p0s.append((x1_l, 0))
                p1s.append((x1_l, H - 1))
            p0s.append((x1, 0))
            p1s.append((x1, H - 1))
            if x1_r <= W - 3:
                p0s.append((x1_r, 0))
                p1s.append((x1_r, H - 1))

            im_sub, mask = add_line(im_sub, mask, p0s, p1s)
        elif abs(y1 - y2) == 1 and (20 < y1 < H - 20):
            expand = np.random.randint(low=10, high=y1 - 9, size=2)
            y1_u = y1 - expand[0]
            y1_b = y1 + expand[1]
            p0s, p1s = [], []
            if y1_u >= 3:
                p0s.append((0, y1_u))
                p1s.append((W - 1, y1_u))
            p0s.append((0, y1))
            p1s.append((W - 1, y1))
            if y1_b <= H - 3:
                p0s.append((0, y1_b))
                p1s.append((W - 1, y1_b))

            im_sub, mask = add_line(im_sub, mask, p0s, p1s)
        elif abs(x1 - x2) > 10 and abs(y1 - y2) > 10:
            k = (y1 - y2) / (x2 - x1)
            b = - k * x1 - y1
            # (y1 - y2)x + (x1 - x2)y + x1y2 -y1x2 = 0
            expand = np.random.randint(low=10, high=100, size=2)
            b_up = b - expand[0]
            b_down = b + expand[1]
            if abs(k) > 1:
                # y=3, y=H-3
                p0s = [(int((-3 - bb) / k), 3) for bb in [b, b_up, b_down]]
                p1s = [(int((-H + 3 - bb) / k), H - 3) for bb in [b, b_up, b_down]]
                im_sub, mask = add_line(im_sub, mask, p0s, p1s)
            elif 1 >= abs(k) > 0.05:
                # x=3, x=W-3
                p0s = [(3, int(-k * 3 - bb)) for bb in [b, b_up, b_down]]
                p1s = [(W - 3, int(-k * (W - 3) - bb)) for bb in [b, b_up, b_down]]
                im_sub, mask = add_line(im_sub, mask, p0s, p1s)

    # blur the image
    prob = np.random.rand()
    if prob < 0.5:
        ksize = np.random.choice([3, 5, 7, 9])
        sigmas = np.arange(0.5, ksize, step=0.5)
        im_sub = cv2.GaussianBlur(im_sub, ksize=(ksize, ksize),
                                  sigmaX=np.random.choice(sigmas),
                                  sigmaY=np.random.choice(sigmas))
    elif 0.5 <= prob <= 0.8:
        #     ksize = np.random.choice([3, 5])
        #     im_sub = cv2.medianBlur(im_sub, ksize=ksize)
        # else:
        im_sub_with_mask = np.concatenate([im_sub, mask[:, :, None]], axis=2)
        im_sub_with_mask = elastic_transform_v2(im_sub_with_mask, im_sub.shape[1] * 2,
                                                im_sub.shape[1] * np.random.randint(low=4, high=8) / 100,
                                                im_sub.shape[1] * np.random.randint(low=4, high=8) / 100)
        im_sub, mask = im_sub_with_mask[:, :, :3], im_sub_with_mask[:, :, 3]

    if mask.sum() < min(mask.shape[:2]) / 2:
        return None, None, None, None

    # add foreigne objects to images
    fg_images = []
    total_count = np.random.randint(2, 5)
    fg_count = np.random.randint(low=2, high=5)
    while True:
        random_filenames = np.random.choice(foreign_fg_filenames,
                                            size=total_count,
                                            replace=True)
        for name in random_filenames:
            if len(fg_images) > total_count:
                break
            fg_im = cv2.imread(name, cv2.IMREAD_UNCHANGED)  # BGRA
            if fg_im.shape[2] != 4:
                continue
            x, y = np.where(fg_im[:, :, -1])
            xmin, ymin, xmax, ymax = np.min(x), np.min(y), np.max(x), np.max(y)
            if xmax - xmin < 10 or ymax - ymin < 10:
                continue
            fg_images.append(fg_im[xmin:xmax, ymin:ymax, :])

        if len(fg_images) > 1:
            break
    print('num_fg_images: ', len(fg_images))
    # paste the fg_image to im_sub
    im_sub1, im_sub1_com, im_sub1_blend, mask1, boxes1, boxes1_mask = \
        paste_fg_images_to_bg(im_sub, mask, fg_images, fg_count)

    mask2 = mask.copy()
    mask2[np.where(boxes1_mask)] = 2

    save_path = os.path.join(save_dir, file_prefix + '_withLineForeign.tif')
    # return im_sub1, mask1, mask2, boxes1
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(save_path, orig_width, orig_height, 3, gdal.GDT_Byte)
    # options=['COMPRESS=LZW', 'BIGTIFF=YES', 'INTERLEAVE=PIXEL'])
    # outdata = driver.CreateCopy(save_path, ds, 0, ['COMPRESS=LZW', 'BIGTIFF=YES', 'INTERLEAVE=PIXEL'])
    outdata.SetGeoTransform(geotransform)  # sets same geotransform as input
    outdata.SetProjection(projection)  # sets same projection as input

    for b in range(3):
        band = outdata.GetRasterBand(b + 1)
        band.WriteArray(im_sub1[:, :, b], xoff=0, yoff=0)
        # band.SetNoDataValue(no_data_value)
        band.FlushCache()
        del band
    outdata.FlushCache()
    del outdata
    del driver



if __name__ == '__main__':
    # test_paste()
    # sys.exit(-1)

    generate_test_images()
    sys.exit(-1)


    if os.name == 'nt':
        bg_images_dir = 'D:/train1_bg_images/'
        save_root = 'E:/line_foreign_object_detection/augmented_data_v2/'
        foreign_fg_dir = 'E:/line_foreign_object_detection/fg_images'
        refine_line_aug_parallel(subset='train', aug_times=1, save_root=save_root,
                                 crop_height=0, crop_width=0,
                                 fg_images_filename=None, fg_boxes_filename=None,
                                 bg_images_dir=bg_images_dir, random_count=2000,
                                 foreign_fg_dir=foreign_fg_dir)
        # bg_images_dir = 'D:/val1_bg_images/'
        refine_line_aug_parallel(subset='val', aug_times=1, save_root=save_root,
                                 crop_height=0, crop_width=0,
                                 fg_images_filename=None, fg_boxes_filename=None,
                                 bg_images_dir=bg_images_dir, random_count=500,
                                 foreign_fg_dir=foreign_fg_dir)
    else:

        bg_images_dir = '/media/ubuntu/Data/gd_cached_path/train1_bg_images/'
        save_root = '/media/ubuntu/Data/line_foreign_object_detection/augmented_data/'
        foreign_fg_dir = '/media/ubuntu/Data/line_foreign_object_detection/fg_images'
        refine_line_aug(subset='train', aug_times=1, save_root=save_root,
                        crop_height=0, crop_width=0,
                        fg_images_filename=None, fg_boxes_filename=None,
                        bg_images_dir=bg_images_dir, random_count=2000,
                        foreign_fg_dir=foreign_fg_dir)
        bg_images_dir = '/media/ubuntu/Data/gd_cached_path/val1_bg_images/'
        refine_line_aug(subset='val', aug_times=1, save_root=save_root,
                        crop_height=0, crop_width=0,
                        fg_images_filename=None, fg_boxes_filename=None,
                        bg_images_dir=bg_images_dir, random_count=1000,
                        foreign_fg_dir=foreign_fg_dir)
