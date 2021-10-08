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
import sklearn
from sklearn.cluster import KMeans

if os.name == 'nt':
    im0_filepath = 'E:/Downloads/BData/images_with_alpha/038_aug_0.png'
    im1_filepath = 'E:/Downloads/BData/images_with_alpha/038_aug_1.png'
    colors_images_dir = 'E:/Downloads/BData/insulator_colors'
else:
    im0_filepath = '/media/ubuntu/Data/insulator/BData/images_with_alpha/038_aug_0.png'
    im1_filepath = '/media/ubuntu/Data/insulator/BData/images_with_alpha/038_aug_1.png'
    colors_images_dir = '/media/ubuntu/Data/insulator/BData/insulator_colors'


def generate_insulator_fg_images(count, debug=False):
    # im0 = cv2.imread('E:/Downloads/BData/images_with_alpha/038_aug_0.png', cv2.IMREAD_UNCHANGED)
    # im1 = cv2.imread('E:/Downloads/BData/images_with_alpha/038_aug_1.png', cv2.IMREAD_UNCHANGED)
    im0 = cv2.imread(im0_filepath, cv2.IMREAD_UNCHANGED)
    im1 = cv2.imread(im1_filepath, cv2.IMREAD_UNCHANGED)

    if debug:
        print(im0.shape, im1.shape)
        print(np.unique(im0[:, :, 3]), np.unique(im1[:, :, 3]))

    fg_images = []
    fg_boxes = []
    for index in range(count):
        boxes = []
        boxes1 = []
        ims = [im1]
        h, w = im1.shape[:2]
        label = [1]
        xmins = [0]
        for step in range(np.random.randint(low=10, high=20)):
            if np.random.rand() < 0.05:
                ims.append(im0)
                boxes1.append([w, 0, w + im0.shape[1], im0.shape[0], 2])  # 2 is defective insulator
                xmins.append(w)
                w += im0.shape[1]
                label.append(0)
            else:
                ims.append(im1)
                xmins.append(w)
                w += im1.shape[1]
                label.append(1)
        ims.append(im1)
        xmins.append(w)
        w += im1.shape[1]
        label.append(1)

        im = np.concatenate(ims, axis=1)
        im[:, :, :3] = cv2.medianBlur(im[:, :, :3], np.random.choice([3, 5, 7, 9]))

        if np.all(label):
            boxes.append([1, 1, w - 1, h - 1, 1])  # 1 is big normal
            if debug:
                print('normal insuator')
            # cv2.imwrite(os.path.join(save_dir, 'normal', '%03d.jpg' % index), im)
            fg_images.append(im)
            fg_boxes.append(boxes)
            continue

        boxes.append([1, 1, w - 1, h - 1, 2])  # 2 is big defective
        if debug:
            print('defective insulator')
        # cv2.imwrite(os.path.join(save_dir, 'defective', '%03d.jpg' % index), im)

        bgr, alpha = im[:, :, :3].copy(), im[:, :, 3:]
        if debug: print(bgr.shape, alpha.shape)
        for box in boxes1:
            xmin, ymin, xmax, ymax, box_label = box
            if debug: print(xmin, ymin, xmax, ymax)
            cv2.rectangle(bgr, (xmin + 1, ymin + 1), (xmax - 1, ymax - 1), color=(0, 0, 255), thickness=2)
            # cv2.imwrite(os.path.join(save_dir, 'shown', '%03d.png' % index),
            #             np.concatenate([bgr, alpha], axis=2))

        # find all the 1->0, then find the right nearest 0->1
        if debug: print('label', label)
        inds = []
        ind = 0
        while ind < len(label) - 1:
            if label[ind] == 1 and label[ind + 1] == 0:
                s1 = ind
                ind2 = ind + 1
                while label[ind2] != 1:
                    ind2 += 1
                s2 = ind2

                inds.append([s1, s2])

                ind = ind2
            else:
                ind += 1
        if debug: print(inds)
        for s1, s2 in inds:
            if debug: print(xmins[s1], 0, xmins[s2], 0)

        bgr, alpha = im[:, :, :3].copy(), im[:, :, 3:]
        if debug: print(bgr.shape, alpha.shape)
        for s1, s2 in inds:
            # xmin, ymin = xmins[s1] + np.random.randint(5, 15), np.random.randint(5, 10)
            # xmax, ymax = xmins[s2] + im0.shape[1] - np.random.randint(5, 15), im0.shape[0] - np.random.randint(5, 10)
            xmin, ymin = xmins[s1], 1
            xmax, ymax = xmins[s2] + im0.shape[1], im0.shape[0]
            cv2.rectangle(bgr, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
            # cv2.imwrite(os.path.join(save_dir, 'shown', '%03d_merged.png' % index),
            #             bgr)
            boxes.append([xmin, ymin, xmax, ymax, 3])

        fg_images.append(im)
        fg_boxes.append(boxes)

        if False and np.random.rand() > 0.5:
            # colors_images = [cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            #                  for filename in glob.glob(os.path.join('E:/Downloads/BData/insulator_colors/*.png'))]
            colors_images = [cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                             for filename in glob.glob(os.path.join(colors_images_dir, '*.png'))]
            color_centers = []
            n_clusters = 20
            for im in colors_images:
                print(im.shape)
                cluster = KMeans(n_clusters=n_clusters)
                cluster.fit(im.reshape(-1, 3))
                print('cluster centers: ', cluster.cluster_centers_)
                new_cluster = []
                for center in cluster.cluster_centers_:
                    if np.any(center < 240):
                        new_cluster.append(center)
                color_centers.append(np.array(new_cluster))
                print(cluster.labels_.shape)
            print(color_centers)
            # sys.exit(-1)

            print('colored')
            predefined_cluster = color_centers[np.random.choice(np.arange(len(color_centers)))].astype(np.uint8)
            im_new = im.copy()
            bgr, alpha = im_new[:, :, :3], im_new[:, :, 3]
            print('alpha unique', np.unique(alpha))
            cluster = KMeans(n_clusters=predefined_cluster.shape[0])
            X = bgr[np.where(alpha)].reshape(-1, 3)
            print(X[:10, :])
            cluster.fit(X)
            labels = cluster.labels_
            print('cluster center', cluster.cluster_centers_)
            new_bgr = np.zeros((len(labels), 3), dtype=np.uint8)
            for c in range(predefined_cluster.shape[0]):
                new_bgr[labels == c, :] = predefined_cluster[np.random.choice(np.arange(predefined_cluster.shape[0]))]
            bgr[np.where(alpha)] = new_bgr + np.random.randint(-30, 30, size=(len(labels), 3))
            bgr[bgr < 0] = 0
            bgr[bgr > 255] = 255
            bgr = cv2.GaussianBlur(bgr, ksize=(np.random.choice([3, 5, 7]), np.random.choice([3, 5, 7])),
                                   sigmaX=np.random.choice([0.1, 1, 10]))
            print('bgr', bgr.shape)
            im_new = np.concatenate([bgr, alpha[:, :, None]], axis=2)
            # cv2.imwrite(os.path.join(save_dir, 'shown', '%03d_colored.png' % index),
            #             np.concatenate([bgr, alpha[:, :, None]], axis=2))

            bgr = bgr.copy()
            for s1, s2 in inds:
                cv2.rectangle(bgr,
                              (xmins[s1] + np.random.randint(5, 15), np.random.randint(5, 10)),
                              (
                                  xmins[s2] + im0.shape[1] - np.random.randint(5, 15),
                                  im0.shape[0] - np.random.randint(5, 10)),
                              color=(0, 0, 255), thickness=2)
                # cv2.imwrite(os.path.join(save_dir, 'shown', '%03d_colored_merged.png' % index),
                #             bgr)

    return fg_images, fg_boxes


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


def paste_fg_images_to_bg(im, mask, mask_k, all_fg_images, all_fg_boxes, count=10):
    im = im.copy()
    mask = mask.copy()
    mask2 = mask.copy()
    mask_k = mask_k.copy()
    boxes = []

    H, W = im.shape[:2]
    boxes_mask = np.zeros((H, W), dtype=np.uint8)
    im_pil = Image.fromarray(im)
    fg_all_pil = Image.new('RGBA', im_pil.size)
    for step in range(count):
        Ys, Xs = np.where(mask2)
        if len(Ys) == 0:
            continue

        # fg = np.random.choice(all_fg_images)
        fg_ind = np.random.randint(len(all_fg_images))
        fg = all_fg_images[fg_ind]
        fg_boxes = all_fg_boxes[fg_ind]
        scale_h = np.random.randint(20, 30) / 200.
        scale_w = scale_h * np.random.randint(80, 100) / 100.
        fg = cv2.resize(fg, dsize=None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_NEAREST)
        h, w = fg.shape[:2]

        # cv2.imwrite('E:/temp1_%d.png'%step, fg)

        fg_boxes = [[max(1, int(scale_w * box[0])),
                     max(1, int(scale_h * box[1])),
                     min(w - 1, int(scale_w * box[2])),
                     min(h - 1, int(scale_h * box[3])), box[-1]] for box in fg_boxes]

        extent = max(h, w) // 2
        indices = np.arange(len(Ys))
        kkkk = 0
        cx, cy = 0, 0
        while True:
            if kkkk > 10:
                break
            index = np.random.choice(indices)
            cy, cx = Ys[index], Xs[index]
            if cy < (extent + 10) // 2 or cx < (extent + 10) // 2 \
                    or H - cy < (extent + 10) // 2 or W - cx < (extent + 10) // 2:
                kkkk += 1
                cx, cy = 0, 0
                continue
            else:
                break

        if cx == 0 or cy == 0:
            continue

        h, w = fg.shape[:2]
        cx, cy = int(cx), int(cy)

        # rotate
        angle = mask_k[cy, cx]
        print('cx, cy, angle, scale_w, scale_h, h, w', cx, cy, type(cx), type(cy), angle, np.degrees(angle), scale_w,
              scale_h, h, w)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((w // 2, h // 2), np.degrees(angle), 1.0)
        print('M', M)
        print('fg', fg.shape)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - w // 2
        M[1, 2] += (nH / 2) - h // 2
        fg = cv2.warpAffine(fg, M, (nW, nH), flags=cv2.INTER_NEAREST)

        h1, w1 = fg.shape[:2]

        print('rotate boxes...')
        fg_boxes_new = []
        for box in fg_boxes:
            print(box)
            x1, y1, x2, y2, box_label = box
            box_im = np.zeros((h, w), dtype=np.uint8)
            box_im[y1:y2, x1:x2] = 128
            box_im_new = cv2.warpAffine(box_im, M, (nW, nH), flags=cv2.INTER_NEAREST)
            points = np.stack(np.where(box_im_new == 128), axis=1)
            y1, x1 = np.min(points, axis=0)
            y2, x2 = np.max(points, axis=0)
            x1, y1 = max(1, x1), max(1, y1)
            x2, y2 = min(w1 - 1, x2), min(h1 - 1, y2)
            fg_boxes_new.append([x1, y1, x2, y2, box_label])

        # fg1 = fg[:, :, :3].copy()
        # for box in fg_boxes_new:
        #     xmin, ymin, xmax, ymax, box_label = box
        #     cv2.rectangle(fg1, (xmin, ymin), (xmax, ymax), color=(255,255,255), thickness=2)
        # cv2.imwrite('E:/temp2_%d.png'%step, fg1)

        bgr = fg[:, :, :3]
        alpha = fg[:, :, 3]
        h, w = fg.shape[:2]

        # w, h = int(scale * fg.shape[1]), int(scale * fg.shape[0])
        print('current insulator', step, w, h)
        # bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)

        # # 改成灰灰的白白的
        # r = np.random.randint(low=190, high=230)
        # g = r - np.random.randint(1, 10)
        # b = g - np.random.randint(1, 10)
        # bgr[:, :, 0] = b
        # bgr[:, :, 1] = g
        # bgr[:, :, 2] = r

        ksize = np.random.choice([3, 5, 7, 9])
        sigmaX = np.random.choice([0.1, 1, 10])
        bgr = cv2.GaussianBlur(bgr, ksize=(ksize, ksize), sigmaX=sigmaX)

        # alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)
        # print(bgr.shape, alpha.shape)

        prob = np.random.rand()
        if prob < 0.2:
            bgr = bgr[:, :, np.random.permutation([0, 1, 2])]
        if 0.2 < prob < 0.5:
            bgr = bgr[:, :, np.random.choice([0, 1, 2], size=3)]

        fg = np.concatenate([bgr, alpha[:, :, None]], axis=-1)

        # cv2.imwrite('E:/temp3_%d.png'%step, fg)

        fg_pil = Image.fromarray(fg).convert('RGBA')

        if cx > 0 and cy > 0:
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            if x1 <= 0 or y1 <= 0 or x2 >= W - 1 or y2 >= H - 1 or np.sum(boxes_mask[y1:y2, x1:x2]) > 0:
                continue

            im_pil.paste(fg_pil, (x1, y1), fg_pil)
            fg_all_pil.paste(fg_pil, (x1, y1), fg_pil)
            boxes_mask[y1:y2, x1:x2] = alpha > 0
            # mask[y1:y2, x1:x2] = 0
            yyy, xxx = np.where(alpha > 0)
            mask[yyy + y1, xxx + x1] = 0
            mask2[y1-5:y2+5, x1-5:x2+5] = 0
            # boxes.append([x1, y1, x2, y2])
            boxes += [[box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1, box[-1]] for box in fg_boxes_new]

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


def add_line_to_image(im, crop_width=0, crop_height=0, subset='train', add_insulators=False):
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
    mask_k = np.zeros((H, W), dtype=np.float32)
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
            mask_k[mask > 0] = np.pi / 4
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
            mask_k[mask > 0] = 0
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
            mask_k[mask > 0] = np.arctan(k)

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
        im_sub_with_mask = np.concatenate([im_sub.astype(np.float32),
                                           mask[:, :, None].astype(np.float32),
                                           mask_k[:, :, None]], axis=2)
        im_sub_with_mask = elastic_transform_v2(im_sub_with_mask, im_sub.shape[1] * 2,
                                                im_sub.shape[1] * np.random.randint(low=4, high=8) / 100,
                                                im_sub.shape[1] * np.random.randint(low=4, high=8) / 100)
        im_sub, mask, mask_k = im_sub_with_mask[:, :, :3].astype(np.uint8), \
                               im_sub_with_mask[:, :, 3].astype(np.uint8), \
                               im_sub_with_mask[:, :, 4]

    if mask.sum() < min(mask.shape[:2]) / 2:
        return None, None, None, None

    # add insulators objects to images
    if add_insulators:
        total_count = np.random.randint(20, 30) if 'train' in subset else 5
        fg_count = np.random.randint(low=2, high=5) if 'train' in subset else np.random.randint(low=2, high=3)
        # while True:
        #     random_filenames = np.random.choice(foreign_fg_filenames,
        #                                         size=total_count,
        #                                         replace=True)
        #     for name in random_filenames:
        #         if len(fg_images) > total_count:
        #             break
        #         fg_im = cv2.imread(name, cv2.IMREAD_UNCHANGED)  # BGRA
        #         if fg_im.shape[2] != 4:
        #             continue
        #         x, y = np.where(fg_im[:, :, -1])
        #         xmin, ymin, xmax, ymax = np.min(x), np.min(y), np.max(x), np.max(y)
        #         if xmax - xmin < 10 or ymax - ymin < 10:
        #             continue
        #         fg_images.append(fg_im[xmin:xmax, ymin:ymax, :])
        #
        #     if len(fg_images) > 1:
        #         break
        all_fg_images, all_fg_boxes = generate_insulator_fg_images(total_count)
        print('num_fg_images: ', len(all_fg_images))
        # paste the fg_image to im_sub
        im_sub1, im_sub1_com, im_sub1_blend, mask1, boxes1, boxes1_mask = \
            paste_fg_images_to_bg(im_sub, mask, mask_k, all_fg_images, all_fg_boxes, fg_count)

        mask2 = mask.copy()
        mask2[np.where(boxes1_mask)] = 2

        return im_sub1, mask1, mask2, boxes1

    return im_sub, mask, None, None


def refine_line_aug(subset='train', aug_times=1, save_root=None,
                    crop_height=512, crop_width=512,
                    fg_images_filename=None, fg_boxes_filename=None,
                    bg_images_dir=None, random_count=1000,
                    add_insulators=True):
    save_dir = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = '%s/images/' % (save_dir)
    images_shown_root = '%s/images_shown/' % (save_dir)
    labels_root = '%s/annotations/' % (save_dir)
    labels_with_insulators_root = '%s/annotations_with_insulators/' % (save_dir)
    for p in [images_root, labels_root, images_shown_root, labels_with_insulators_root]:
        if not os.path.exists(p):
            os.makedirs(p)

    aug_times = aug_times if 'train' in subset else 1
    bg_filenames = glob.glob(bg_images_dir + '/*.png')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["good", "bad", "defect"]):  # 1: good, 2:bad, 3:defective
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    palette = np.array([[0, 0, 0], [255, 255, 255], [255, 255, 0], [0, 255, 255], [0, 0, 255]])

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
                                                          add_insulators=add_insulators)

            if im1 is None:
                continue
            if mask1.sum() < min(im1.shape[:2]) / 2:
                continue

            save_prefix = '%s_%d_%010d' % (file_prefix, aug_time, bg_ind)
            cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1[:, :, ::-1])  # 不能有中文
            cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)
            cv2.imwrite('%s/%s.png' % (labels_with_insulators_root, save_prefix), mask2)

            lines.append('%s\n' % save_prefix)

            if len(boxes1) > 0:
                sub_h, sub_w = im1.shape[:2]
                # save image
                # for coco format
                single_image = {'file_name': save_prefix + '.jpg', 'id': image_id, 'width': sub_w, 'height': sub_h}
                data_dict['images'].append(single_image)

                mask2_color = np.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    mask2_color[mask2 == label, :] = color

                im2 = im1.copy()
                for box2 in boxes1:
                    xmin, ymin, xmax, ymax, box_label = box2

                    if True:
                        cv2.rectangle(im2, (xmin, ymin), (xmax, ymax), color=palette[box_label].tolist(),
                                      thickness=2)

                    xc1 = int((xmin + xmax) / 2)
                    yc1 = int((ymin + ymax) / 2)
                    w1 = xmax - xmin
                    h1 = ymax - ymin
                    # for coco format
                    single_obj = {'area': int(w1 * h1),
                                  'category_id': box_label,
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


def process_one(bg_ind, bg_filenames, subset, add_insulators, aug_time, images_root, labels_root,
                labels_with_insulators_root, images_shown_root):
    bg_filename = bg_filenames[bg_ind]
    file_prefix = bg_filename.split(os.sep)[-1].replace('.png', '')
    bg = cv2.imread(bg_filename)
    palette = np.array([[0, 0, 0], [255, 255, 255], [255, 255, 0], [0, 255, 255], [0, 0, 255]])

    if min(bg.shape[:2]) < 512:
        return None, None, None

    im1, mask1, mask2, boxes1 = add_line_to_image(bg, crop_height=0, crop_width=0, subset=subset,
                                                  add_insulators=add_insulators)

    if im1 is None:
        return None, None, None
    if mask1.sum() < min(im1.shape[:2]) / 2:
        return None, None, None

    save_prefix = '%s_%d_%010d' % (file_prefix, aug_time, bg_ind)
    cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1[:, :, ::-1])  # 不能有中文
    cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)
    cv2.imwrite('%s/%s.png' % (labels_with_insulators_root, save_prefix), mask2)

    mask2_color = np.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        mask2_color[mask2 == label, :] = color

    for box2 in boxes1:
        xmin, ymin, xmax, ymax, box_label = box2
        cv2.rectangle(im1, (xmin, ymin), (xmax, ymax), color=palette[box_label].tolist(), thickness=2)
    cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                np.concatenate([im1[:, :, ::-1],
                                255 * np.stack([mask1, mask1, mask1], axis=2),
                                mask2_color], axis=1))  # 不能有中文

    return save_prefix, im1.shape[:2], boxes1


def refine_line_aug_parallel(subset='train', aug_times=1, save_root=None,
                             crop_height=512, crop_width=512,
                             fg_images_filename=None, fg_boxes_filename=None,
                             bg_images_dir=None, random_count=1000,
                             add_insulators=True):
    save_dir = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = '%s/images/' % (save_dir)
    images_shown_root = '%s/images_shown/' % (save_dir)
    labels_root = '%s/annotations/' % (save_dir)
    labels_with_insulators_root = '%s/annotations_with_insulators/' % (save_dir)
    for p in [images_root, labels_root, images_shown_root, labels_with_insulators_root]:
        if not os.path.exists(p):
            os.makedirs(p)

    aug_times = aug_times if 'train' in subset else 1
    bg_filenames = glob.glob(bg_images_dir + '/*.png')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(["good", "bad", "defect"]):  # 1,2,3 is gan, 4 is jueyuanzi
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
            itertools.repeat(add_insulators),
            itertools.repeat(aug_time),
            itertools.repeat(images_root),
            itertools.repeat(labels_root),
            itertools.repeat(labels_with_insulators_root),
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
                    xmin, ymin, xmax, ymax, box_label = box2

                    xc1 = int((xmin + xmax) / 2)
                    yc1 = int((ymin + ymax) / 2)
                    w1 = xmax - xmin
                    h1 = ymax - ymin
                    # for coco format
                    single_obj = {'area': int(w1 * h1),
                                  'category_id': box_label,
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
    save_dir = '/media/ubuntu/Data/insulator/check_paster_insulator'
    im_dir = '/media/ubuntu/Data/gd_newAug5_Rot0_4classes_bak/refine_line_v1_512_512/train/images'
    mask_dir = '/media/ubuntu/Data/gd_newAug5_Rot0_4classes_bak/refine_line_v1_512_512/train/annotations'
    im_filenames = glob.glob(os.path.join(im_dir, '*.jpg'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for step in range(10):
        filename = np.random.choice(im_filenames)
        file_prefix = filename.split(os.sep)[-1].replace('.jpg', '')
        im_sub = cv2.imread(filename)
        mask = cv2.imread(os.path.join(mask_dir, file_prefix + '.png'))[:, :, 0]

        # add insulator objects to images
        total_count = 100
        all_fg_images, all_fg_boxes = generate_insulator_fg_images(total_count)
        print('num_fg_images: ', len(all_fg_images))
        import pdb
        pdb.set_trace()
        # paste the fg_image to im_sub
        mask_k = mask * 90
        im_sub1, im_sub1_com, im_sub1_blend, mask1, boxes1, boxes1_mask = \
            paste_fg_images_to_bg(im_sub, mask, mask_k, all_fg_images, all_fg_boxes)

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


if __name__ == '__main__':
    # test_paste()
    # sys.exit(-1)

    if os.name == 'nt':
        bg_images_dir = 'D:/train1_bg_images/'
        save_root = 'E:/insulator_detection/augmented_data_v2/'
        refine_line_aug(subset='train', aug_times=1, save_root=save_root,
                                 crop_height=0, crop_width=0,
                                 fg_images_filename=None, fg_boxes_filename=None,
                                 bg_images_dir=bg_images_dir, random_count=2000,
                                 add_insulators=True)
        # bg_images_dir = 'D:/val1_bg_images/'
        # refine_line_aug(subset='val', aug_times=1, save_root=save_root,
        #                 crop_height=0, crop_width=0,
        #                 fg_images_filename=None, fg_boxes_filename=None,
        #                 bg_images_dir=bg_images_dir, random_count=500,
        #                 add_insulators=True)
    else:

        bg_images_dir = '/media/ubuntu/Data/gd_cached_path/train1_bg_images/'
        save_root = '/media/ubuntu/Data/line_foreign_object_detection/augmented_data/'
        refine_line_aug(subset='train', aug_times=1, save_root=save_root,
                        crop_height=0, crop_width=0,
                        fg_images_filename=None, fg_boxes_filename=None,
                        bg_images_dir=bg_images_dir, random_count=2000,
                        add_insulators=True)
        bg_images_dir = '/media/ubuntu/Data/gd_cached_path/val1_bg_images/'
        refine_line_aug(subset='val', aug_times=1, save_root=save_root,
                        crop_height=0, crop_width=0,
                        fg_images_filename=None, fg_boxes_filename=None,
                        bg_images_dir=bg_images_dir, random_count=1000,
                        add_insulators=True)
