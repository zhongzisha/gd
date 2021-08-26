import sys,os,glob,shutil,time
import argparse
import socket
import numpy as np
import cv2
from osgeo import gdal, ogr, osr
from natsort import natsorted
from myutils import elastic_transform_v2, load_gt_for_detection, load_gt_polys_from_esri_xml, \
    compute_offsets


def add_line(im, mask, p0s, p1s):
    line_width = np.random.randint(1, 3)
    for p0, p1 in zip(p0s, p1s):

        d = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        if d < 30:
            continue

        r = np.random.randint(160, 220)
        g = r - np.random.randint(1, 10)
        b = g - np.random.randint(1, 10)

        cv2.line(im, p0, p1, color=(r, g, b), thickness=line_width)
        cv2.line(mask, p0, p1, color=(1, 0, 0), thickness=line_width)

    return im, mask


def add_line_to_image(im, crop_width=0, crop_height=0):
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
            return None, None
        H, W = crop_height, crop_width
    else:
        im_sub = im

    mask = np.zeros((H, W), dtype=np.uint8)
    for step in range(np.random.randint(2, 5)):
        y = np.random.randint(0, H - 1, size=2)
        x = np.random.randint(0, W - 1, size=2)

        x1, x2 = x
        y1, y2 = y
        if abs(x1 - x2) == 1 and (20 < x1 < W - 20):
            expand = np.random.randint(low=10, high=x1-9, size=2)
            x1_l = x1 - expand[0]
            x1_r = x1 + expand[1]
            p0s, p1s = [], []
            if x1_l >= 3:
                p0s.append((x1_l, 0))
                p1s.append((x1_l, H-1))
            p0s.append((x1, 0))
            p1s.append((x1, H-1))
            if x1_r <= W-3:
                p0s.append((x1_r, 0))
                p1s.append((x1_r, H-1))

            im_sub, mask = add_line(im_sub, mask, p0s, p1s)
        elif abs(y1 - y2) == 1 and (20 < y1 < H - 20):
            expand = np.random.randint(low=10, high=y1-9, size=2)
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
                p1s.append((W-1, y1_b))

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
                p0s = [(int((-3 - bb)/k), 3) for bb in [b, b_up, b_down]]
                p1s = [(int((-H+3 - bb)/k), H-3) for bb in [b, b_up, b_down]]
                im_sub, mask = add_line(im_sub, mask, p0s, p1s)
            elif 1 >= abs(k) > 0.05:
                # x=3, x=W-3
                p0s = [(3, int(-k*3-bb)) for bb in [b, b_up, b_down]]
                p1s = [(W-3, int(-k*(W-3)-bb)) for bb in [b, b_up, b_down]]
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

    return im_sub, mask


def refine_line_aug(subset='train', aug_times=1, save_root=None,
                    crop_height=512, crop_width=512,
                    bg_images_dir=None, random_count=1000):
    save_dir = '%s/%s/' % (save_root, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = '%s/images/' % (save_dir)
    images_shown_root = '%s/images_shown/' % (save_dir)
    labels_root = '%s/annotations/' % (save_dir)
    for p in [images_root, labels_root, images_shown_root]:
        if not os.path.exists(p):
            os.makedirs(p)

    aug_times = aug_times if 'train' in subset else 1
    bg_filenames = glob.glob(bg_images_dir + '/*.png')

    lines = []
    for aug_time in range(aug_times):
        if len(bg_filenames) > random_count:
            bg_indices = np.random.choice(np.arange(len(bg_filenames)), size=random_count, replace=False)
        else:
            bg_indices = np.arange(len(bg_filenames))

        for bg_ind in bg_indices:
            bg_filename = bg_filenames[bg_ind]
            file_prefix = bg_filename.split(os.sep)[-1].replace('.png', '')
            bg = cv2.imread(bg_filename)

            if min(bg.shape[:2]) < 512:
                continue

            im1, mask1 = add_line_to_image(bg, crop_height, crop_width)

            if im1 is None:
                continue
            if mask1.sum() < min(im1.shape[:2]) / 2:
                continue

            save_prefix = '%s_%d_%010d' % (file_prefix, aug_time, bg_ind)
            cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), im1)  # 不能有中文
            cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), mask1)

            lines.append('%s\n' % save_prefix)

            if True: #np.random.rand() < 0.01:
                cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix),
                            np.concatenate([im1, 255 * np.stack([mask1, mask1, mask1], axis=2)],
                                           axis=1))  # 不能有中文
    if len(lines) > 0:
        with open('%s/%s.txt' % (save_root, subset), 'w') as fp:
            fp.writelines(lines)


def extract_bg_images(source, gt_dir, subset='train', save_root=None, random_count=0, update_cache=False):
    hostname = socket.gethostname()
    # if hostname == 'master':
    #     source = '/media/ubuntu/Data/%s_list.txt' % (subset)
    #     gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial'
    # else:
    #     source = 'E:/%s_list.txt' % (subset)  # sys.argv[1]
    #     gt_dir = 'F:/gddata/aerial'  # sys.argv[2]

    save_dir = '%s/%s_bg_images/' % (save_root, subset)

    if not update_cache and os.path.exists(save_dir) and len(glob.glob(save_dir + '/*.png')) > 0:
        return save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

            gt_boxes, gt_boxes_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
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

                if np.min(im1) == np.max(im1):
                    continue

                if len(np.where(im1[:, :, 0] > 0)[0]) < 0.5 * np.prod(im1.shape[:2]):
                    continue

                cv2.imencode('.png', im1)[1].tofile('%s/bg_%d_%d.png' % (save_dir, ti, j))

            del mask
    return save_dir



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
    parser.add_argument('--bg_images_dir', default='', type=str)
    parser.add_argument('--source', default='', type=str)
    parser.add_argument('--gt_dir', default='', type=str)

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

    bg_images_dir = args.bg_images_dir
    if bg_images_dir == '':
        print('error')
        sys.exit(-1)

    source = args.source
    gt_dir = args.gt_dir

    # for detection aug
    if hostname == 'master':
        save_root = '/media/ubuntu/Data/gd_newAug%d_Rot%d_4classes_line' % (aug_times, do_rotate)
    else:
        save_root = 'E:/gd_newAug%d_Rot%d_4classes' % (aug_times, do_rotate)

    print('extract bg images ...')
    bg_images_dir = extract_bg_images(source, gt_dir, subset, cached_data_path, random_count, update_cache=update_cache)

    if aug_type == 'refine_line_v1':
        # TODO zzs need to generate the parallel lines like the wire

        # save_root/refine_line_v1_512_512/train/images/*.jpg
        # save_root/refine_line_v1_512_512/train/annotations/*.png
        # save_root/refine_line_v1_512_512/train/train.txt
        # save_root/refine_line_v1_512_512/val/images/*.jpg
        # save_root/refine_line_v1_512_512/val/annotations/*.png
        # save_root/refine_line_v1_512_512/val/val.txt
        save_root = '%s/%s_%d_%d' % (save_root, aug_type, crop_height, crop_width)
        refine_line_aug(subset=subset, aug_times=aug_times, save_root=save_root,
                        crop_height=crop_height, crop_width=crop_width,
                        bg_images_dir=bg_images_dir, random_count=random_count)
    else:
        print('wrong aug type')
        sys.exit(-1)