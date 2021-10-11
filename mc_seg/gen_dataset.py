import sys, os, glob, shutil

sys.path.insert(0, 'F:/gd')
import cv2
import numpy as np
from osgeo import gdal, osr, ogr
from natsort import natsorted
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, box_iou_np, \
    box_intersection_np, load_gt_polys_from_esri_xml, compute_offsets, alpha_map, elastic_transform_v2, \
    load_gt_for_detection
import json
from PIL import Image
import time
import xml.dom.minidom


def main(subset):
    save_dir = 'E:\\Downloads\\mc_seg\\data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = save_dir + "/images/%s/" % subset
    labels_root = save_dir + "/annotations/%s/" % subset
    images_shown_root = save_dir + "/images_shown/%s/" % subset
    if not os.path.exists(images_root):
        os.makedirs(images_root)
    if not os.path.exists(labels_root):
        os.makedirs(labels_root)
    if not os.path.exists(images_shown_root):
        os.makedirs(images_shown_root)

    # RGB
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]])
    opacity = 0.5

    lines = []
    size0 = 10000
    size1 = -1

    subsizes = [4096, 2048, 1024]
    scales = [0.25, 0.5, 1.0]
    gt_postfixes = [
        '_newgt_landslide',
        '_newgt_water',
        '_newgt_tree',
        '_newgt_building'
    ]
    gt_dir = 'E:\\Downloads\\mc_seg\\xmls'

    tiffile = 'G:\\gddata\\all\\2-WV03-在建杆塔.tif'
    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

    print(file_prefix)
    label_maps = {
        1: 'landslide',
        2: 'water',
        3: 'tree',
        4: 'building',
    }
    names_to_labels_map = {v: k for k, v in label_maps.items()}
    # 'landslide', 'tree', 'water', 'building'

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
    all_gt_polys, all_gt_labels = [], []
    for gi, gt_postfix in enumerate(gt_postfixes):
        gt_xml_filename = os.path.join(gt_dir, file_prefix + gt_postfix + '.xml')

        # gt_polys, gt_labels = load_gt_polys_from_esri_xml(gt_xml_filename, gdal_trans_info=geotransform,
        #                                                   mapcoords2pixelcoords=mapcoords2pixelcoords)
        DomTree = xml.dom.minidom.parse(gt_xml_filename)
        annotation = DomTree.documentElement
        regionlist = annotation.getElementsByTagName('Region')
        gt_polys = []
        gt_labels = []

        for region in regionlist:
            name = region.getAttribute("name")
            label = name.split('_')[0]
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
                gt_labels.append(names_to_labels_map[label])

        # gt_labels = [gi + 1 for _ in range(len(gt_labels))]
        all_gt_polys.append(gt_polys)
        all_gt_labels.append(gt_labels)

        print('class-%d' % (gi + 1), len(gt_polys), len(gt_labels))

    # import pdb
    # pdb.set_trace()

    # 首先根据标注生成mask图像，存在内存问题！！！
    print('generate mask ...')
    mask = 255 * np.ones((orig_height, orig_width), dtype=np.uint8)
    if True:
        # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
        # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)

        for gt_polys, gt_labels in zip(all_gt_polys, all_gt_labels):
            for poly, label in zip(gt_polys, gt_labels):  # poly为nx2的点, numpy.array
                cv2.drawContours(mask, [poly], -1, color=(label, label, label), thickness=-1)

        mask_savefilename = save_dir + "/" + file_prefix + ".png"
        # cv2.imwrite(mask_savefilename, mask)
        if not os.path.exists(mask_savefilename):
            cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

    time.sleep(3)

    print('mask shape', mask.shape)
    print('crop images ... ')
    indices_y, indices_x = np.where(mask < 255)
    print('indices_y', indices_y, indices_y.shape)
    print('indices_x', indices_x, indices_x.shape)
    count = min(1000 if 'train' in subset else 200, len(indices_y))
    indices = np.random.choice(np.arange(len(indices_y)), size=count, replace=False)
    print('indices', indices)
    for ind in indices:
        print('ind', ind)
        cy, cx = indices_y[ind], indices_x[ind]
        print(cx, cy)
        xmin, ymin = max(1, cx - 512), max(1, cy - 512)
        xmax, ymax = min(orig_width - 1, cx + 512), min(orig_height - 1, cy + 512)
        print(xmin, ymin, xmax, ymax)

        xoffset, yoffset = int(xmin), int(ymin)
        sub_width, sub_height = int(xmax - xmin), int(ymax - ymin)
        print(xoffset, yoffset, sub_width, sub_height)
        img = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
        for b in range(3):
            band = ds.GetRasterBand(b + 1)
            img[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)
        img_sum = np.sum(img, axis=2)
        indices_y1, indices_x1 = np.where(img_sum > 0)
        if len(indices_x1) == 0:
            continue

        # sample points from mask
        seg = mask[(yoffset):(yoffset + sub_height), (xoffset):(xoffset + sub_width)]
        seg_count = len(np.where(seg < 255)[0])
        if seg_count < 10:
            continue

        save_prefix = '%06d' % ind
        cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), img[:, :, ::-1])  # 不能有中文
        cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), seg)

        lines.append('%s\n' % save_prefix)

        if True:
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color

            img = img * (1 - opacity) + color_seg * opacity
            img = img.astype(np.uint8)
            cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix), img[:, :, ::-1])

    if len(lines) > 0:
        with open(save_dir + '/%s.txt' % (subset), 'w') as fp:
            fp.writelines(lines)


def main_using_shp_single_tif(subset):
    shp_root_dir = 'E:\\Downloads\\mc_seg\\shps'
    tiffile = 'G:\\gddata\\all\\2-WV03-在建杆塔.tif'
    tiffile = 'G:\\gddata\\all\\3-wv02-在建杆塔.tif'
    tiffile = 'G:\\gddata\\all\\WV03-曲花甲线-20170510.tif'
    tiffile = 'E:\\gddata_resampled\\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif'
    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
    print(file_prefix)

    save_dir = os.path.join('E:\\Downloads\\mc_seg\\data_using_shp', file_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = save_dir + "/images/%s/" % subset
    labels_root = save_dir + "/annotations/%s/" % subset
    images_shown_root = save_dir + "/images_shown/%s/" % subset
    if not os.path.exists(images_root):
        os.makedirs(images_root)
    if not os.path.exists(labels_root):
        os.makedirs(labels_root)
    if not os.path.exists(images_shown_root):
        os.makedirs(images_shown_root)

    # RGB
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]])
    opacity = 0.5

    lines = []
    label_maps = {
        1: 'landslide',
        2: 'water',
        3: 'tree',
        4: 'building',
    }
    names_to_labels = {v: k for k, v in label_maps.items()}
    # 'landslide', 'tree', 'water', 'building'

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
    all_gt_polys, all_gt_labels = [], []
    for label, label_name in label_maps.items():
        shp_filename = os.path.join(shp_root_dir, file_prefix, label_name + '.shp')
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        shp_ds = shp_driver.Open(shp_filename, update=0)
        layerCount = shp_ds.GetLayerCount()
        layer = shp_ds.GetLayer(0)
        layerDef = layer.GetLayerDefn()
        for i in range(layerDef.GetFieldCount()):
            fieldDef = layerDef.GetFieldDefn(i)
            fieldName = fieldDef.GetName()
            fieldTypeCode = fieldDef.GetType()
            fieldType = fieldDef.GetFieldTypeName(fieldTypeCode)
            fieldWidth = fieldDef.GetWidth()
            fieldPrecision = fieldDef.GetPrecision()

            print(i, fieldName, fieldType, fieldWidth, fieldPrecision)

        gt_polys = []
        gt_labels = []
        for i, feat in enumerate(layer):
            geom = feat.GetGeometryRef()
            label = names_to_labels[feat.GetField(0)]

            wkt_str = geom.ExportToWkt()
            start = wkt_str.find("((") + 2
            end = wkt_str.find("))")
            points = []
            for point in wkt_str[start:end].split(','):
                x, y = point.split(' ')
                points += [float(x), float(y)]
            points = np.array(points).reshape([-1, 2])  # nx2
            if mapcoords2pixelcoords:
                points[:, 0] -= xOrigin
                points[:, 1] -= yOrigin
                points[:, 0] /= pixelWidth
                points[:, 1] /= pixelHeight
                points += 0.5
                points = points.astype(np.int32)

            gt_polys.append(points)
            gt_labels.append(int(float(label)))  # 0 is gan, 1 is jueyuanzi
        shp_ds = None
        all_gt_polys.append(gt_polys)
        all_gt_labels.append(gt_labels)
        print('%s' % label_name, len(gt_polys), len(gt_labels))

    # import pdb
    # pdb.set_trace()

    # 首先根据标注生成mask图像，存在内存问题！！！
    print('generate mask ...')
    mask = 255 * np.ones((orig_height, orig_width), dtype=np.uint8)
    if True:
        # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
        # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)

        for gt_polys, gt_labels in zip(all_gt_polys, all_gt_labels):
            for poly, label in zip(gt_polys, gt_labels):  # poly为nx2的点, numpy.array
                cv2.drawContours(mask, [poly], -1, color=(label, label, label), thickness=-1)

        mask_savefilename = save_dir + "/" + file_prefix + ".png"
        # cv2.imwrite(mask_savefilename, mask)
        if not os.path.exists(mask_savefilename):
            cv2.imencode('.png', mask)[1].tofile(mask_savefilename)

    time.sleep(3)

    print('mask shape', mask.shape)
    print('crop images ... ')
    indices_y, indices_x = np.where(mask < 255)
    print('indices_y', indices_y, indices_y.shape)
    print('indices_x', indices_x, indices_x.shape)
    count = min(100 if 'train' in subset else 10, len(indices_y))
    indices = np.random.choice(np.arange(len(indices_y)), size=count, replace=False)
    print('indices', indices)
    for ind in indices:
        print('ind', ind)
        cy, cx = indices_y[ind], indices_x[ind]
        print(cx, cy)
        xmin, ymin = max(1, cx - 512), max(1, cy - 512)
        xmax, ymax = min(orig_width - 1, cx + 512), min(orig_height - 1, cy + 512)
        print(xmin, ymin, xmax, ymax)

        xoffset, yoffset = int(xmin), int(ymin)
        sub_width, sub_height = int(xmax - xmin), int(ymax - ymin)
        print(xoffset, yoffset, sub_width, sub_height)
        img = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
        for b in range(3):
            band = ds.GetRasterBand(b + 1)
            img[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)
        img_sum = np.sum(img, axis=2)
        indices_y1, indices_x1 = np.where(img_sum > 0)
        if len(indices_x1) == 0:
            continue

        # sample points from mask
        seg = mask[(yoffset):(yoffset + sub_height), (xoffset):(xoffset + sub_width)]
        seg_count = len(np.where(seg < 255)[0])
        if seg_count < 10:
            continue

        save_prefix = '%06d' % ind
        # cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), img[:, :, ::-1])  # 不能有中文
        # cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), seg)
        Image.fromarray(img).save('%s/%s.jpg' % (images_root, save_prefix))
        Image.fromarray(seg).save('%s/%s.png' % (labels_root, save_prefix))

        lines.append('%s\n' % save_prefix)

        if True:
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color

            img = img * (1 - opacity) + color_seg * opacity
            img = img.astype(np.uint8)
            # cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix), img[:, :, ::-1])
            Image.fromarray(img).save('%s/%s.jpg' % (images_shown_root, save_prefix))

    if len(lines) > 0:
        with open(save_dir + '/%s.txt' % (subset), 'w') as fp:
            fp.writelines(lines)


def main_using_shp_multi_tif(subset, random_count=0, use_resampled_tif=False):
    source = 'G:\\gddata\\all'
    source_resampled = 'E:\\gddata_resampled_half'
    shp_root_dir = 'E:\\Downloads\\mc_seg\\shps'
    save_dir = 'E:\\Downloads\\mc_seg\\data_using_shp_multi_random%d' % random_count
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = save_dir + "/images/%s/" % subset
    labels_root = save_dir + "/annotations/%s/" % subset
    images_shown_root = save_dir + "/images_shown/%s/" % subset
    if not os.path.exists(images_root):
        os.makedirs(images_root)
    if not os.path.exists(labels_root):
        os.makedirs(labels_root)
    if not os.path.exists(images_shown_root):
        os.makedirs(images_shown_root)

    # RGB
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]])
    opacity = 0.5

    all_lines = []
    count = 1
    label_maps = {
        1: 'landslide',
        2: 'water',
        3: 'tree',
        4: 'building',
    }
    names_to_labels = {v: k for k, v in label_maps.items()}
    # 'landslide', 'tree', 'water', 'building'

    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(os.path.join(source, '*.tif')))
    print(tiffiles)

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]  # 'E:\\gddata_resampled\\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif'
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue

        # if we have resmapled dataset
        if use_resampled_tif and os.path.exists(os.path.join(source_resampled, file_prefix + '.tif')):
            tiffile = os.path.join(source_resampled, file_prefix + '.tif')

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
        mapcoords2pixelcoords = True
        print('loading gt ...')
        all_gt_polys, all_gt_labels = [], []
        for label, label_name in label_maps.items():
            shp_filename = os.path.join(shp_root_dir, file_prefix, label_name + '.shp')
            if not os.path.exists(shp_filename):
                continue
            shp_driver = ogr.GetDriverByName("ESRI Shapefile")
            shp_ds = shp_driver.Open(shp_filename, update=0)
            layerCount = shp_ds.GetLayerCount()
            layer = shp_ds.GetLayer(0)
            layerDef = layer.GetLayerDefn()
            for i in range(layerDef.GetFieldCount()):
                fieldDef = layerDef.GetFieldDefn(i)
                fieldName = fieldDef.GetName()
                fieldTypeCode = fieldDef.GetType()
                fieldType = fieldDef.GetFieldTypeName(fieldTypeCode)
                fieldWidth = fieldDef.GetWidth()
                fieldPrecision = fieldDef.GetPrecision()

                print(i, fieldName, fieldType, fieldWidth, fieldPrecision)

            gt_polys = []
            gt_labels = []
            for i, feat in enumerate(layer):
                geom = feat.GetGeometryRef()
                label = names_to_labels[feat.GetField(0)]

                wkt_str = geom.ExportToWkt()
                start = wkt_str.find("((") + 2
                end = wkt_str.find("))")
                points = []
                for point in wkt_str[start:end].split(','):
                    x, y = point.split(' ')
                    points += [float(x), float(y)]
                points = np.array(points).reshape([-1, 2])  # nx2
                if mapcoords2pixelcoords:
                    points[:, 0] -= xOrigin
                    points[:, 1] -= yOrigin
                    points[:, 0] /= pixelWidth
                    points[:, 1] /= pixelHeight
                    points += 0.5
                    points = points.astype(np.int32)

                gt_polys.append(points)
                gt_labels.append(int(float(label)))  # 0 is gan, 1 is jueyuanzi
            shp_ds = None
            all_gt_polys.append(gt_polys)
            all_gt_labels.append(gt_labels)
            print('%s' % label_name, len(gt_polys), len(gt_labels))

        # import pdb
        # pdb.set_trace()
        if len(all_gt_polys) == 0:
            continue

        # 首先根据标注生成mask图像，存在内存问题！！！
        print('generate mask ...')
        mask = 255 * np.ones((orig_height, orig_width), dtype=np.uint8)
        gt_indices = []
        if True:
            # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
            # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)

            for gt_polys, gt_labels in zip(all_gt_polys, all_gt_labels):
                for poly, label in zip(gt_polys, gt_labels):  # poly为nx2的点, numpy.array
                    cv2.drawContours(mask, [poly], -1, color=(label, label, label), thickness=-1)

                    xmin, ymin = np.min(poly, axis=0)
                    xmax, ymax = np.max(poly, axis=0)
                    h, w = ymax - ymin, xmax - xmin
                    gt_indices.append(
                        np.stack([
                            np.random.randint(low=ymin, high=ymax, size=4 if 'train' in subset else 1),
                            np.random.randint(low=xmin, high=xmax, size=4 if 'train' in subset else 1)
                        ], axis=-1)
                    )

            mask_savefilename = save_dir + "/" + file_prefix + ".png"
            # cv2.imwrite(mask_savefilename, mask)
            if not os.path.exists(mask_savefilename):
                cv2.imencode('.png', mask)[1].tofile(mask_savefilename)
        time.sleep(1)
        gt_indices = np.concatenate(gt_indices, axis=0)
        gt_indices = np.unique(gt_indices, axis=0)

        print('mask shape', mask.shape)
        print('crop images ... ')
        if random_count > 0:
            indices_y, indices_x = np.where(mask < 255)
            random_count = min(random_count if 'train' in subset else random_count // 4, len(indices_y) // 2)
            indices = np.random.choice(np.arange(len(indices_y)), size=random_count, replace=False)
        else:
            indices_y, indices_x = gt_indices[:, 0], gt_indices[:, 1]
            indices = np.arange(len(indices_y))
        print('indices_y', indices_y, indices_y.shape)
        print('indices_x', indices_x, indices_x.shape)

        print('indices', indices)
        for ind in indices:
            print('ind', ind)
            cy, cx = indices_y[ind], indices_x[ind]
            print(cx, cy)
            xmin, ymin = max(1, cx - 512), max(1, cy - 512)
            xmax, ymax = min(orig_width - 1, cx + 512), min(orig_height - 1, cy + 512)
            print(xmin, ymin, xmax, ymax)

            xoffset, yoffset = int(xmin), int(ymin)
            sub_width, sub_height = int(xmax - xmin), int(ymax - ymin)
            if sub_width < 768 or sub_height < 768:
                continue

            print(xoffset, yoffset, sub_width, sub_height)
            img = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
            for b in [0, 1, 2] if file_prefix not in bands_info else [2, 1, 0]:
                band = ds.GetRasterBand(b + 1)
                img[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)
            img_sum = np.sum(img, axis=2)
            indices_y1, indices_x1 = np.where(img_sum > 0)
            if len(indices_x1) == 0:
                continue

            # sample points from mask
            seg = mask[(yoffset):(yoffset + sub_height), (xoffset):(xoffset + sub_width)]
            seg_count = len(np.where(seg < 255)[0])
            if seg_count < 10:
                continue

            save_prefix = '%09d' % count
            # cv2.imwrite('%s/%s.jpg' % (images_root, save_prefix), img[:, :, ::-1])  # 不能有中文
            # cv2.imwrite('%s/%s.png' % (labels_root, save_prefix), seg)
            Image.fromarray(img).save('%s/%s.jpg' % (images_root, save_prefix))
            Image.fromarray(seg).save('%s/%s.png' % (labels_root, save_prefix))

            all_lines.append('%s\n' % save_prefix)

            if True:
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(palette):
                    color_seg[seg == label, :] = color

                img = img * (1 - opacity) + color_seg * opacity
                img = img.astype(np.uint8)
                # cv2.imwrite('%s/%s.jpg' % (images_shown_root, save_prefix), img[:, :, ::-1])
                Image.fromarray(img).save('%s/%s.jpg' % (images_shown_root, save_prefix))

            count += 1

    if len(all_lines) > 0:
        with open(save_dir + '/%s.txt' % (subset), 'w') as fp:
            fp.writelines(all_lines)


def extract_towers_using_shp_multi_tif(subset, resample_method='nearest', res_in_cm=10):
    source = 'G:\\gddata\\all'
    source_resampled = 'E:\\gddata_resampled_%s_%dcm' % (resample_method, res_in_cm)
    shp_root_dir = 'E:\\gddata_processed'
    save_dir = 'E:\\Downloads\\tower_detection\\data_using_shp_multi_%s_res%dcm' % (resample_method, res_in_cm)
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_root = save_dir + "/images/%s/" % subset
    labels_root = save_dir + "/annotations/%s/" % subset
    images_shown_root = save_dir + "/images_shown/%s/" % subset
    if not os.path.exists(images_root):
        os.makedirs(images_root)
    if not os.path.exists(labels_root):
        os.makedirs(labels_root)
    if not os.path.exists(images_shown_root):
        os.makedirs(images_shown_root)

    # RGB
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255]])
    opacity = 0.5

    all_lines = []
    count = 1
    label_maps = {
        # 1: 'landslide',
        # 2: 'water',
        # 3: 'tree',
        # 4: 'building',
        5: 'tower'
    }
    names_to_labels = {v: k for k, v in label_maps.items()}
    # 'landslide', 'tree', 'water', 'building'

    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(os.path.join(source, '*.tif')))
    print(tiffiles)

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]  # 'E:\\gddata_resampled\\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif'
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue

        # if we have resampled dataset
        if os.path.exists(os.path.join(source_resampled, file_prefix + '.tif')):
            tiffile = os.path.join(source_resampled, file_prefix + '.tif')

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
        mapcoords2pixelcoords = True
        print('loading gt ...')
        all_gt_polys, all_gt_labels = [], []
        for label, label_name in label_maps.items():
            shp_filename = os.path.join(shp_root_dir, file_prefix, label_name + '.shp')
            if not os.path.exists(shp_filename):
                continue
            shp_driver = ogr.GetDriverByName("ESRI Shapefile")
            shp_ds = shp_driver.Open(shp_filename, update=0)
            layerCount = shp_ds.GetLayerCount()
            layer = shp_ds.GetLayer(0)
            layerDef = layer.GetLayerDefn()
            for i in range(layerDef.GetFieldCount()):
                fieldDef = layerDef.GetFieldDefn(i)
                fieldName = fieldDef.GetName()
                fieldTypeCode = fieldDef.GetType()
                fieldType = fieldDef.GetFieldTypeName(fieldTypeCode)
                fieldWidth = fieldDef.GetWidth()
                fieldPrecision = fieldDef.GetPrecision()

                print(i, fieldName, fieldType, fieldWidth, fieldPrecision)

            gt_polys = []
            gt_labels = []
            for i, feat in enumerate(layer):
                geom = feat.GetGeometryRef()
                label = names_to_labels[feat.GetField(0)]

                wkt_str = geom.ExportToWkt()
                start = wkt_str.find("((") + 2
                end = wkt_str.find("))")
                points = []
                for point in wkt_str[start:end].split(','):
                    x, y = point.split(' ')
                    points += [float(x), float(y)]
                points = np.array(points).reshape([-1, 2])  # nx2
                if mapcoords2pixelcoords:
                    points[:, 0] -= xOrigin
                    points[:, 1] -= yOrigin
                    points[:, 0] /= pixelWidth
                    points[:, 1] /= pixelHeight
                    points += 0.5
                    points = points.astype(np.int32)

                gt_polys.append(points)
                gt_labels.append(int(float(label)))  # 0 is gan, 1 is jueyuanzi
            shp_ds = None
            all_gt_polys.append(gt_polys)
            all_gt_labels.append(gt_labels)
            print('%s' % label_name, len(gt_polys), len(gt_labels))

            for pi, poly in enumerate(gt_polys):
                xmin, ymin = np.min(poly, axis=0)
                xmax, ymax = np.max(poly, axis=0)
                h, w = ymax - ymin, xmax - xmin
                xoffset, yoffset, sub_width, sub_height = int(xmin), int(ymin), int(w), int(h)
                print(xoffset, yoffset, sub_width, sub_height)

                img = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
                for b in [0, 1, 2] if file_prefix not in bands_info else [2, 1, 0]:
                    band = ds.GetRasterBand(b + 1)
                    img[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)

                save_prefix = '%s_%s_%03d' % (file_prefix, label_name, pi)
                Image.fromarray(img).save('%s/%s.jpg' % (images_root, save_prefix))
                all_lines.append('%s\n' % save_prefix)

    if len(all_lines) > 0:
        with open(save_dir + '/%s.txt' % (subset), 'w') as fp:
            fp.writelines(all_lines)


def convert_envi_xml_to_separate_shapefile():
    src_dir = r"G:\gddata\all"
    xml_dir = r"E:\Downloads\mc_seg\xmls"
    save_dir = r"E:\Downloads\mc_seg\shps"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_filenames = glob.glob(os.path.join(src_dir, '*.tif'))

    label_maps = {
        1: 'landslide',
        2: 'water',
        3: 'tree',
        4: 'building',
    }
    names_to_labels_map = {v: k for k, v in label_maps.items()}
    # 'landslide', 'water', 'tree', 'building'
    label_postfixes = [
        '_newgt_landslide',
        '_newgt_water',
        '_newgt_tree',
        '_newgt_building'
    ]

    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    for filename in input_filenames:
        file_prefix = filename.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix:
            continue

        ds = gdal.Open(filename, gdal.GA_ReadOnly)

        gdal_trans_info = ds.GetGeoTransform()
        gdal_projection = ds.GetProjection()
        gdal_projection_sr = osr.SpatialReference(wkt=gdal_projection)

        ds = None

        polys = []
        labels = []
        # change xml filename
        for xml_postfix in label_postfixes:
            xml_filename = os.path.join(xml_dir, file_prefix + xml_postfix + '.xml')
            if not os.path.exists(xml_filename):
                continue
            print(xml_filename)
            DomTree = xml.dom.minidom.parse(xml_filename)
            annotation = DomTree.documentElement
            regionlist = annotation.getElementsByTagName('Region')
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
                    points = np.array(coords).reshape([-1, 2])  # nx2
                    # if mapcoords2pixelcoords:    # geo coordinates to pixel coordinates
                    #     points[:, 0] -= xOrigin
                    #     points[:, 1] -= yOrigin
                    #     points[:, 0] /= pixelWidth
                    #     points[:, 1] /= pixelHeight
                    #     points += 0.5
                    #     points = points.astype(np.int32)

                    polys.append(points)
                    labels.append(label)

        if len(labels) > 0:
            labels = [names_to_labels_map[label] for label in labels]
            labels = np.array(labels)
            for label in np.unique(labels):
                if label not in label_maps.keys():
                    continue
                inds = np.where(labels == label)[0]
                label_name = label_maps[label]
                print(label, label_name, len(inds))

                shp_dir = os.path.join(save_dir, file_prefix)
                if not os.path.exists(shp_dir):
                    os.makedirs(shp_dir)

                shp_filename = os.path.join(shp_dir, label_name + '.shp')
                outDataSource = outDriver.CreateDataSource(shp_filename)
                outLayer = outDataSource.CreateLayer(label_name, gdal_projection_sr, geom_type=ogr.wkbPolygon)
                featureDefn = outLayer.GetLayerDefn()
                newField = ogr.FieldDefn("Class", ogr.OFTString)
                outLayer.CreateField(newField)
                for ind in inds:
                    coords = polys[ind]

                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    for xx, yy in coords:
                        ring.AddPoint(xx, yy)
                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)

                    # add new geom to layer
                    outFeature = ogr.Feature(featureDefn)
                    outFeature.SetGeometry(poly)
                    outFeature.SetField("Class", label_name)
                    outLayer.CreateFeature(outFeature)
                    outFeature.Destroy()

                featureDefn = None
                outLayer = None
                outDataSource.Destroy()
                outDataSource = None

        # break


def convert_envi_tower_xml_to_seperate_shapefile():
    src_dir = r'G:\gddata\all'
    save_dir = r'E:\gddata_processed'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    input_filenames = glob.glob(os.path.join(src_dir, '*.tif'))

    label_maps = {
        3: 'tower'
    }
    label_postfixes = ['_gt_5']

    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    for filename in input_filenames:
        file_prefix = filename.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix:
            continue

        ds = gdal.Open(filename, gdal.GA_ReadOnly)

        gdal_trans_info = ds.GetGeoTransform()
        gdal_projection = ds.GetProjection()
        gdal_projection_sr = osr.SpatialReference(wkt=gdal_projection)

        ds = None

        polys = []
        labels = []
        # change xml filename
        for xml_postfix in label_postfixes:
            xml_filename = os.path.join(src_dir, file_prefix + xml_postfix + '.xml')
            if not os.path.exists(xml_filename):
                continue
            print(xml_filename)
            DomTree = xml.dom.minidom.parse(xml_filename)
            annotation = DomTree.documentElement
            regionlist = annotation.getElementsByTagName('Region')
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
                    points = np.array(coords).reshape([-1, 2])  # nx2
                    # if mapcoords2pixelcoords:    # geo coordinates to pixel coordinates
                    #     points[:, 0] -= xOrigin
                    #     points[:, 1] -= yOrigin
                    #     points[:, 0] /= pixelWidth
                    #     points[:, 1] /= pixelHeight
                    #     points += 0.5
                    #     points = points.astype(np.int32)

                    polys.append(points)
                    labels.append(int(float(label)))

        if len(labels) > 0:
            labels = np.array(labels)
            for label in np.unique(labels):
                if label not in label_maps.keys():
                    continue
                inds = np.where(labels == label)[0]
                label_name = label_maps[label]
                print(label, label_name, len(inds))

                shp_dir = os.path.join(save_dir, file_prefix)
                if not os.path.exists(shp_dir):
                    os.makedirs(shp_dir)

                shp_filename = os.path.join(shp_dir, label_name + '.shp')
                outDataSource = outDriver.CreateDataSource(shp_filename)
                outLayer = outDataSource.CreateLayer(label_name, gdal_projection_sr, geom_type=ogr.wkbPolygon)
                featureDefn = outLayer.GetLayerDefn()
                newField = ogr.FieldDefn("Class", ogr.OFTString)
                outLayer.CreateField(newField)
                for ind in inds:
                    coords = polys[ind]

                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    for xx, yy in coords:
                        ring.AddPoint(xx, yy)
                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)

                    # add new geom to layer
                    outFeature = ogr.Feature(featureDefn)
                    outFeature.SetGeometry(poly)
                    outFeature.SetField("Class", label_name)
                    outLayer.CreateFeature(outFeature)
                    outFeature.Destroy()

                featureDefn = None
                outLayer = None
                outDataSource.Destroy()
                outDataSource = None

        # break


def resample_aerial_tifs(method='cubic', res_in_cm=10):
    dst_dir = 'E:\\gddata_resampled_%s_%dcm' % (method, res_in_cm)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in glob.glob(os.path.join("E:\\gddata_resampled_cubic_30cm\\*.tif")):
        fileprefix = filename.split(os.sep)[-1].replace('.tif', '')
        command = r'gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=LZW" -co "BIGTIFF=YES" -tr %f %f -r ' \
                  r'%s %s %s' % (
                      res_in_cm / 100., res_in_cm / 100., method,
                      os.path.join(r"G:\\gddata\\all", fileprefix + '.tif'),
                      os.path.join(dst_dir, fileprefix + '.tif')
                  )
        os.system(command)
        time.sleep(2)
        print(filename)


def gen_tower_detection_dataset():
    source = 'G:/gddata/all'  # 'E:/%s_list.txt' % subset  # sys.argv[1]
    gt_dir = 'G:/gddata/all'  # sys.argv[2]
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"

    save_dir = 'E:/Downloads/tower_detection_noGap/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    valid_labels_set = [3, 4]
    label_maps = {
        3: 'tower',
        4: 'insulator'
    }
    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    for p in [save_img_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    data_dict = {'images': [], 'categories': [], 'annotations': []}
    train_dict = {'images': [], 'categories': [], 'annotations': []}
    val_dict = {'images': [], 'categories': [], 'annotations': []}
    for idx, name in enumerate(["tower0", "tower1", "tower", "insulator"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)
        train_dict['categories'].append(single_cat)
        val_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

    subsizes = [1024]
    scales = [1.0]
    gaps = [0]  # [256]

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue

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

        print('loading gt ...')
        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                    valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        for si, (subsize, scale) in enumerate(zip(subsizes, scales)):

            for gap in gaps:
                offsets = compute_offsets(height=orig_height, width=orig_width, subsize=subsize, gap=gap)

                for oi, (xoffset, yoffset, sub_w, sub_h) in enumerate(offsets):  # left, up
                    # sub_width = min(orig_width, big_subsize)
                    # sub_height = min(orig_height, big_subsize)
                    # if xoffset + sub_width > orig_width:
                    #     sub_width = orig_width - xoffset
                    # if yoffset + sub_height > orig_height:
                    #     sub_height = orig_height - yoffset
                    print(oi, len(offsets), xoffset, yoffset, sub_w, sub_h)
                    save_prefix = '%d_%d_%d_%d' % (ti, si, oi, gap)

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
                    for bi in [0, 1, 2] if file_prefix not in bands_info else [2, 1, 0]:
                        band = ds.GetRasterBand(bi + 1)
                        band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_w, win_ysize=sub_h)
                        cutout.append(band_data)
                    cutout = np.stack(cutout, -1)  # RGB

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
                            if area1 >= 0.7 * area:
                                sub_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                            else:
                                invalid_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                    # if len(np.where(im1[:, :, 0] > 0)[0]) < 0.5 * np.prod(im1.shape[:2]):
                    #     continue

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

                    prob = np.random.rand()
                    if len(sub_gt_boxes) > 0:

                        single_image = {'file_name': save_prefix + '.jpg', 'id': image_id, 'width': sub_w,
                                        'height': sub_h}
                        data_dict['images'].append(single_image)
                        if prob > 0.2:
                            train_dict['images'].append(single_image)
                        else:
                            val_dict['images'].append(single_image)

                        cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                        save_img = True
                        for box2 in sub_gt_boxes:
                            xmin, ymin, xmax, ymax, label = box2

                            if save_img:
                                cv2.rectangle(cutout, (xmin, ymin), (xmax, ymax), color=colors[label],
                                              thickness=3)

                            xc1 = int((xmin + xmax) / 2)
                            yc1 = int((ymin + ymax) / 2)
                            w1 = xmax - xmin
                            h1 = ymax - ymin
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
                            if prob > 0.2:
                                train_dict['annotations'].append(single_obj)
                            else:
                                val_dict['annotations'].append(single_obj)

                            inst_count = inst_count + 1

                        image_id = image_id + 1

                        if save_img:
                            cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR
                    # else:
                    #     # if no gt_boxes
                    #     if len(np.where(cutout[:, :, 0] > 0)[0]) < 0.5 * np.prod(cutout.shape[:2]):
                    #         continue
                    #
                    #     if np.random.rand() < 0.1:
                    #         # for yolo format
                    #         save_prefix = save_prefix + '_noGT'
                    #         cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                    # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                    # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                    # ims.append(im)

    if inst_count > 1:
        with open(save_dir + '/all.json', 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)
        with open(save_dir + '/train.json', 'w') as f_out:
            json.dump(train_dict, f_out, indent=4)
        with open(save_dir + '/val.json', 'w') as f_out:
            json.dump(val_dict, f_out, indent=4)


def gen_tower_detection_dataset_v3():
    source = 'G:/gddata/all'  # 'E:/%s_list.txt' % subset  # sys.argv[1]
    gt_dir = 'G:/gddata/all'  # sys.argv[2]
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"

    save_dir = 'E:/Downloads/tower_detection_v3_1/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    valid_labels_set = [3, 4]
    label_maps = {
        3: 'tower',
        4: 'insulator'
    }
    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    for p in [save_img_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    data_dict = {'images': [], 'categories': [], 'annotations': []}
    train_dict = {'images': [], 'categories': [], 'annotations': []}
    val_dict = {'images': [], 'categories': [], 'annotations': []}
    for idx, name in enumerate(["tower0", "tower1", "tower", "insulator"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)
        train_dict['categories'].append(single_cat)
        val_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

    subsizes = [1024]
    scales = [1.0]
    gaps = [256]  # [256]

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue

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

        print('loading gt ...')
        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                    valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        # gt_boxes1 = np.copy(gt_boxes)
        # xmin, ymin = np.min(gt_boxes1[:, 0:2], axis=0)
        # xmax, ymax = np.max(gt_boxes1[:, 2:4], axis=0)
        # w, h = xmax - xmin, ymax - ymin
        #
        # box_centers = []
        # for box in gt_boxes1:
        #     x1, y1, x2, y2 = box
        #     xc = (x1 + x2) // 2
        #     yc = (y1 + y2) // 2
        #     box_centers.append([xc, yc])
        # box_centers = np.array(box_centers).reshape(-1, 2)
        #
        # if w > h:
        #     idx = np.argsort(box_centers[:, 0])
        # else:
        #     idx = np.argsort(box_centers[:, 1])
        #
        # gt_boxes1 = gt_boxes1[idx]
        # box_centers = box_centers[idx]
        # idx = int(np.floor(0.8 * len(box_centers)))
        # if idx == box_centers.shape[0] - 1:
        #     idx = box_centers.shape[0] - 2
        # split_xc = (box_centers[idx, 0] + box_centers[idx + 1, 0]) // 2
        # split_yc = (box_centers[idx, 1] + box_centers[idx + 1, 1]) // 2
        # if w > h:
        #     print('using split_xc')
        # else:
        #     print('using split_yc')

        for si, (subsize, scale) in enumerate(zip(subsizes, scales)):

            for gap in gaps:
                offsets = compute_offsets(height=orig_height, width=orig_width, subsize=subsize, gap=gap)

                for oi, (xoffset, yoffset, sub_w, sub_h) in enumerate(offsets):  # left, up
                    # sub_width = min(orig_width, big_subsize)
                    # sub_height = min(orig_height, big_subsize)
                    # if xoffset + sub_width > orig_width:
                    #     sub_width = orig_width - xoffset
                    # if yoffset + sub_height > orig_height:
                    #     sub_height = orig_height - yoffset
                    print(oi, len(offsets), xoffset, yoffset, sub_w, sub_h)
                    save_prefix = '%d_%d_%d_%d' % (ti, si, oi, gap)

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
                    for bi in [0, 1, 2] if file_prefix not in bands_info else [2, 1, 0]:
                        band = ds.GetRasterBand(bi + 1)
                        band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_w, win_ysize=sub_h)
                        cutout.append(band_data)
                    cutout = np.stack(cutout, -1)  # RGB

                    if np.min(cutout[:, :, 0]) == np.max(cutout[:, :, 0]):
                        continue

                    # check is in train or in val
                    # if w > h:
                    #     print('using split_xc')
                    #     if split_xc > (xmax1 + sub_w // 2):
                    #         is_train = True
                    #     else:
                    #         is_train = False
                    # else:
                    #     print('using split_yc')
                    #     if split_yc > (ymax1 + sub_h // 2):
                    #         is_train = True
                    #     else:
                    #         is_train = False
                    if np.random.rand() < 0.8:
                        is_train = True
                    else:
                        is_train = False

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
                            if area1 >= 0.7 * area:
                                sub_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                            else:
                                invalid_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                    # if len(np.where(im1[:, :, 0] > 0)[0]) < 0.5 * np.prod(im1.shape[:2]):
                    #     continue

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

                    if len(sub_gt_boxes) > 0:

                        for degree in [0, 90, 180, 270] if is_train else [0]:
                            cutout_new, sub_gt_boxes_new = rotate_image(cutout, sub_gt_boxes, degree=degree)
                            save_prefix = '%s_%d' % (save_prefix, degree)
                            sub_h, sub_w = cutout_new.shape[:2]
                            single_image = {'file_name': save_prefix + '.jpg', 'id': image_id, 'width': sub_w,
                                            'height': sub_h}
                            data_dict['images'].append(single_image)
                            if is_train:
                                train_dict['images'].append(single_image)
                            else:
                                val_dict['images'].append(single_image)

                            cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout_new[:, :, ::-1])  # RGB --> BGR

                            save_img = True
                            for box2 in sub_gt_boxes_new:
                                xmin, ymin, xmax, ymax, label = box2

                                if save_img:
                                    cv2.rectangle(cutout_new, (xmin, ymin), (xmax, ymax), color=colors[label],
                                                  thickness=3)

                                xc1 = int((xmin + xmax) / 2)
                                yc1 = int((ymin + ymax) / 2)
                                w1 = xmax - xmin
                                h1 = ymax - ymin
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
                                if is_train:
                                    train_dict['annotations'].append(single_obj)
                                else:
                                    val_dict['annotations'].append(single_obj)

                                inst_count = inst_count + 1

                            image_id = image_id + 1

                            if save_img:
                                cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout_new[:, :, ::-1])  # RGB --> BGR
                    # else:
                    #     # if no gt_boxes
                    #     if len(np.where(cutout[:, :, 0] > 0)[0]) < 0.5 * np.prod(cutout.shape[:2]):
                    #         continue
                    #
                    #     if np.random.rand() < 0.1:
                    #         # for yolo format
                    #         save_prefix = save_prefix + '_noGT'
                    #         cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                    # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                    # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                    # ims.append(im)

    if inst_count > 1:
        with open(save_dir + '/all.json', 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)
        with open(save_dir + '/train.json', 'w') as f_out:
            json.dump(train_dict, f_out, indent=4)
        with open(save_dir + '/val.json', 'w') as f_out:
            json.dump(val_dict, f_out, indent=4)


def gen_tower_detection_dataset_v4(aug_times=1, subset=None):
    if subset is None:
        source = 'G:/gddata/all'  # 'E:/%s_list.txt' % subset  # sys.argv[1]
    else:
        source = 'E:/Downloads/tower_detection/%s_list.txt' % subset  # sys.argv[1]
    gt_dir = 'G:/gddata/all'  # sys.argv[2]
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"

    if subset is None:
        save_dir = 'E:/Downloads/tower_detection/v4_augtimes%d_800_200_noSplit/' % aug_times
    else:
        save_dir = 'E:/Downloads/tower_detection/v4_augtimes%d_800_200/%s' % (aug_times, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(source):
        shutil.copy(source, os.path.join(save_dir, os.path.basename(source)))

    valid_labels_set = [3, 4]
    label_maps = {
        3: 'tower',
        4: 'insulator'
    }
    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    for p in [save_img_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    data_dict = {'images': [], 'categories': [], 'annotations': []}
    train_dict, val_dict = None, None
    if subset is None:
        train_dict = {'images': [], 'categories': [], 'annotations': []}
        val_dict = {'images': [], 'categories': [], 'annotations': []}
    for idx, name in enumerate(["tower0", "tower1", "tower", "insulator"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)
        if subset is None:
            train_dict['categories'].append(single_cat)
            val_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

    subsizes = [800]  # [1024]
    scales = [1.0]
    gaps = [200]  # [256]  # [256]

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]

        if not os.path.exists(tiffile):
            continue

        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue

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

        print('loading gt ...')
        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                    valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        is_train = True
        if subset is not None:
            if 'train' in subset:
                is_train = True
            else:
                is_train = False

        for si, (subsize, scale) in enumerate(zip(subsizes, scales)):

            for gap in gaps:
                offsets = compute_offsets(height=orig_height, width=orig_width, subsize=subsize, gap=gap)

                for oi, (xoffset0, yoffset0, sub_w0, sub_h0) in enumerate(offsets):  # left, up
                    # sub_width = min(orig_width, big_subsize)
                    # sub_height = min(orig_height, big_subsize)
                    # if xoffset + sub_width > orig_width:
                    #     sub_width = orig_width - xoffset
                    # if yoffset + sub_height > orig_height:
                    #     sub_height = orig_height - yoffset
                    if subset is None:
                        if np.random.rand() < 0.8:
                            is_train = True
                        else:
                            is_train = False

                    print(oi, len(offsets), xoffset0, yoffset0, sub_w0, sub_h0)

                    for aug_time in range(aug_times) if is_train else [0]:
                        xoffset, yoffset, sub_w, sub_h = xoffset0, yoffset0, sub_w0, sub_h0
                        save_prefix = '%d_%d_%d_%d_%d' % (ti, si, oi, gap, aug_time)
                        if aug_time > 0:
                            xoffset += np.random.randint(-0.2*sub_w, 0.2*sub_w)
                            yoffset += np.random.randint(-0.2*sub_h, 0.2*sub_h)

                        xoffset = max(1, xoffset)
                        yoffset = max(1, yoffset)
                        if xoffset + sub_w > orig_width - 1:
                            sub_w = orig_width - 1 - xoffset
                        if yoffset + sub_h > orig_height - 1:
                            sub_h = orig_height - 1 - yoffset

                        if sub_w < 512 or sub_h < 512:
                            continue

                        xoffset, yoffset, sub_w, sub_h = [int(val) for val in
                                                          [xoffset, yoffset, sub_w, sub_h]]
                        xmin1, ymin1 = xoffset, yoffset
                        xmax1, ymax1 = xoffset + sub_w, yoffset + sub_h

                        cutout = []
                        for bi in [0, 1, 2] if file_prefix not in bands_info else [2, 1, 0]:
                            band = ds.GetRasterBand(bi + 1)
                            band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_w, win_ysize=sub_h)
                            cutout.append(band_data)
                        cutout = np.stack(cutout, -1)  # RGB

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
                                if area1 >= 0.4 * area:
                                    sub_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                                else:
                                    invalid_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                        # if len(np.where(im1[:, :, 0] > 0)[0]) < 0.5 * np.prod(im1.shape[:2]):
                        #     continue

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

                        if len(sub_gt_boxes) > 0:

                            for degree in [0, 90, 180, 270] if is_train else [0]:
                                cutout_new, sub_gt_boxes_new = rotate_image(cutout.copy(), sub_gt_boxes, degree=degree)
                                save_prefix = '%s_%d' % (save_prefix, degree)
                                sub_h, sub_w = cutout_new.shape[:2]
                                single_image = {'file_name': save_prefix + '.jpg', 'id': image_id, 'width': sub_w,
                                                'height': sub_h}

                                data_dict['images'].append(single_image)
                                if subset is None:
                                    if is_train:
                                        train_dict['images'].append(single_image)
                                    else:
                                        val_dict['images'].append(single_image)

                                cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout_new[:, :, ::-1])  # RGB --> BGR

                                save_img = True
                                for box2 in sub_gt_boxes_new:
                                    xmin, ymin, xmax, ymax, label = box2

                                    if save_img:
                                        cv2.rectangle(cutout_new, (xmin, ymin), (xmax, ymax), color=colors[label],
                                                      thickness=3)

                                    xc1 = int((xmin + xmax) / 2)
                                    yc1 = int((ymin + ymax) / 2)
                                    w1 = xmax - xmin
                                    h1 = ymax - ymin
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
                                    if subset is None:
                                        if is_train:
                                            train_dict['annotations'].append(single_obj)
                                        else:
                                            val_dict['annotations'].append(single_obj)

                                    inst_count = inst_count + 1

                                image_id = image_id + 1

                                if save_img:
                                    cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout_new[:, :, ::-1])
                                    # RGB --> BGR
                        # else:
                        #     # if no gt_boxes
                        #     if len(np.where(cutout[:, :, 0] > 0)[0]) < 0.5 * np.prod(cutout.shape[:2]):
                        #         continue
                        #
                        #     if np.random.rand() < 0.1:
                        #         # for yolo format
                        #         save_prefix = save_prefix + '_noGT'
                        #         cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                        # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                        # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                        # ims.append(im)

    if inst_count > 1:
        if subset is None:
            with open(save_dir + '/all.json', 'w') as f_out:
                json.dump(data_dict, f_out, indent=4)
            with open(save_dir + '/train.json', 'w') as f_out:
                json.dump(train_dict, f_out, indent=4)
            with open(save_dir + '/val.json', 'w') as f_out:
                json.dump(val_dict, f_out, indent=4)
        else:
            with open(save_dir + '/%s.json' % subset, 'w') as f_out:
                json.dump(data_dict, f_out, indent=4)


# with histogram matching
def gen_tower_detection_dataset_v5(aug_times=1, subset=None):
    if subset is None:
        source = 'G:/gddata/all'  # 'E:/%s_list.txt' % subset  # sys.argv[1]
    else:
        source = 'E:/Downloads/tower_detection/%s_list.txt' % subset  # sys.argv[1]
    gt_dir = 'G:/gddata/all'  # sys.argv[2]
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"

    if subset is None:
        save_dir = 'E:/Downloads/tower_detection/v5_augtimes%d_800_200_noSplit/' % aug_times
    else:
        save_dir = 'E:/Downloads/tower_detection/v5_augtimes%d_800_200/%s' % (aug_times, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(source):
        shutil.copy(source, os.path.join(save_dir, os.path.basename(source)))

    valid_labels_set = [3, 4]
    label_maps = {
        3: 'tower',
        4: 'insulator'
    }
    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    for p in [save_img_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    data_dict = {'images': [], 'categories': [], 'annotations': []}
    train_dict, val_dict = None, None
    if subset is None:
        train_dict = {'images': [], 'categories': [], 'annotations': []}
        val_dict = {'images': [], 'categories': [], 'annotations': []}
    for idx, name in enumerate(["tower0", "tower1", "tower", "insulator"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)
        if subset is None:
            train_dict['categories'].append(single_cat)
            val_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

    subsizes = [800]  # [1024]
    scales = [1.0]
    gaps = [200]  # [256]  # [256]

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]

        if not os.path.exists(tiffile):
            continue

        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue

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

        print('loading gt ...')
        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                    valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        is_train = True
        if subset is not None:
            if 'train' in subset:
                is_train = True
            else:
                is_train = False

        for si, (subsize, scale) in enumerate(zip(subsizes, scales)):

            for gap in gaps:
                offsets = compute_offsets(height=orig_height, width=orig_width, subsize=subsize, gap=gap)

                for oi, (xoffset0, yoffset0, sub_w0, sub_h0) in enumerate(offsets):  # left, up
                    # sub_width = min(orig_width, big_subsize)
                    # sub_height = min(orig_height, big_subsize)
                    # if xoffset + sub_width > orig_width:
                    #     sub_width = orig_width - xoffset
                    # if yoffset + sub_height > orig_height:
                    #     sub_height = orig_height - yoffset
                    if subset is None:
                        if np.random.rand() < 0.8:
                            is_train = True
                        else:
                            is_train = False

                    print(oi, len(offsets), xoffset0, yoffset0, sub_w0, sub_h0)

                    for aug_time in range(aug_times) if is_train else [0]:
                        xoffset, yoffset, sub_w, sub_h = xoffset0, yoffset0, sub_w0, sub_h0

                        if aug_time > 0:
                            xoffset += np.random.randint(-0.2*sub_w, 0.2*sub_w)
                            yoffset += np.random.randint(-0.2*sub_h, 0.2*sub_h)

                        xoffset = max(1, xoffset)
                        yoffset = max(1, yoffset)
                        if xoffset + sub_w > orig_width - 1:
                            sub_w = orig_width - 1 - xoffset
                        if yoffset + sub_h > orig_height - 1:
                            sub_h = orig_height - 1 - yoffset

                        if sub_w < 512 or sub_h < 512:
                            continue

                        xoffset, yoffset, sub_w, sub_h = [int(val) for val in
                                                          [xoffset, yoffset, sub_w, sub_h]]
                        xmin1, ymin1 = xoffset, yoffset
                        xmax1, ymax1 = xoffset + sub_w, yoffset + sub_h

                        cutout = []
                        for bi in [0, 1, 2] if file_prefix not in bands_info else [2, 1, 0]:
                            band = ds.GetRasterBand(bi + 1)
                            band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_w, win_ysize=sub_h)
                            cutout.append(band_data)
                        cutout = np.stack(cutout, -1)  # RGB

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
                                if area1 >= 0.4 * area:
                                    sub_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                                else:
                                    invalid_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                        # if len(np.where(im1[:, :, 0] > 0)[0]) < 0.5 * np.prod(im1.shape[:2]):
                        #     continue

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

                        if len(sub_gt_boxes) > 0:
                            cutout_sum = np.sum(cutout, axis=2)
                            if pixelHeight < 0.5 and \
                                    (len(np.where(cutout_sum == 0)[0]) < 30
                                     or len(np.where(cutout_sum == 255*3)[0])):
                                prefix = 'high'
                            else:
                                prefix = 'low'

                            save_prefix = '%s_%d_%d_%d_%d_%d' % (prefix, ti, si, oi, gap, aug_time)
                            for degree in [0, 90, 180, 270] if is_train else [0]:
                                cutout_new, sub_gt_boxes_new = rotate_image(cutout.copy(), sub_gt_boxes, degree=degree)
                                save_prefix = '%s_%d' % (save_prefix, degree)
                                sub_h, sub_w = cutout_new.shape[:2]
                                single_image = {'file_name': save_prefix + '.jpg', 'id': image_id, 'width': sub_w,
                                                'height': sub_h}

                                data_dict['images'].append(single_image)
                                if subset is None:
                                    if is_train:
                                        train_dict['images'].append(single_image)
                                    else:
                                        val_dict['images'].append(single_image)

                                cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout_new[:, :, ::-1])  # RGB --> BGR

                                save_img = True
                                for box2 in sub_gt_boxes_new:
                                    xmin, ymin, xmax, ymax, label = box2

                                    if save_img:
                                        cv2.rectangle(cutout_new, (xmin, ymin), (xmax, ymax), color=colors[label],
                                                      thickness=3)

                                    xc1 = int((xmin + xmax) / 2)
                                    yc1 = int((ymin + ymax) / 2)
                                    w1 = xmax - xmin
                                    h1 = ymax - ymin
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
                                    if subset is None:
                                        if is_train:
                                            train_dict['annotations'].append(single_obj)
                                        else:
                                            val_dict['annotations'].append(single_obj)

                                    inst_count = inst_count + 1

                                image_id = image_id + 1

                                if save_img:
                                    cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout_new[:, :, ::-1])
                                    # RGB --> BGR
                        # else:
                        #     # if no gt_boxes
                        #     if len(np.where(cutout[:, :, 0] > 0)[0]) < 0.5 * np.prod(cutout.shape[:2]):
                        #         continue
                        #
                        #     if np.random.rand() < 0.1:
                        #         # for yolo format
                        #         save_prefix = save_prefix + '_noGT'
                        #         cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                        # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                        # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                        # ims.append(im)

    if subset is None or 'train' in subset:
        train_json_filename = save_dir + '/train_no_hm.json'
        with open(train_json_filename, 'w') as f_out:
            json.dump(train_dict, f_out, indent=4)

        from pycocotools.coco import COCO
        from skimage import exposure
        coco = COCO(train_json_filename)
        img_maps = {v['file_name'].replace('.jpg', ''): k for k, v in coco.imgs.items()}
        good_filenames = glob.glob(os.path.join(save_img_path, 'good_*.jpg'))
        indices = np.arange(len(good_filenames))
        for j in range(1000):
            i = np.random.choice(indices, replace=False, size=2)
            filename1 = good_filenames[i[0]]
            filename2 = good_filenames[i[1]]
            prefix1 = os.path.basename(filename1).replace('.jpg', '')
            prefix2 = os.path.basename(filename2).replace('.jpg', '')
            img1 = cv2.imread(filename1)
            img2 = cv2.imread(filename2)
            multi = True if img1.shape[-1] > 1 else False
            matched1 = exposure.match_histograms(img1, img2, multichannel=multi)
            # matched2 = exposure.match_histograms(img2, img1, multichannel=multi)

            # cv2.imwrite(os.path.join(save_dir, '%s_%s.jpg' % (prefix1, prefix2)),
            #             np.concatenate([img1, img2, matched1, matched2], axis=1))

            img_id = img_maps[prefix1]
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            anns = coco.load_anns(ann_ids)
            sub_gt_boxes_new = []
            for ann in anns:
                x1, y1, bw, bh = ann['bbox']
                sub_gt_boxes_new.append([x1, y1, x1+bw, y1+bh, ann['category_id']])

            save_prefix = '%s_hm' % prefix1
            sub_h, sub_w = matched1.shape[:2]
            single_image = {'file_name': save_prefix + '.jpg', 'id': image_id, 'width': sub_w,
                            'height': sub_h}

            data_dict['images'].append(single_image)
            train_dict['images'].append(single_image)

            cv2.imwrite(save_img_path + save_prefix + '.jpg', matched1)  # RGB --> BGR

            save_img = True
            for box2 in sub_gt_boxes_new:
                xmin, ymin, xmax, ymax, label = box2

                if save_img:
                    cv2.rectangle(matched1, (xmin, ymin), (xmax, ymax), color=colors[label],
                                  thickness=3)

                xc1 = int((xmin + xmax) / 2)
                yc1 = int((ymin + ymax) / 2)
                w1 = xmax - xmin
                h1 = ymax - ymin
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
                train_dict['annotations'].append(single_obj)

                inst_count = inst_count + 1

            image_id = image_id + 1

            if save_img:
                cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', matched1)

    if inst_count > 1:
        if subset is None:
            with open(save_dir + '/all.json', 'w') as f_out:
                json.dump(data_dict, f_out, indent=4)
            with open(save_dir + '/train.json', 'w') as f_out:
                json.dump(train_dict, f_out, indent=4)
            with open(save_dir + '/val.json', 'w') as f_out:
                json.dump(val_dict, f_out, indent=4)
        else:
            with open(save_dir + '/%s.json' % subset, 'w') as f_out:
                json.dump(data_dict, f_out, indent=4)


# using predefined split train and val set
def gen_tower_detection_dataset_v2(subset='train'):
    source = 'G:/gddata/all'  # 'E:\\Downloads\\mc_seg\\tifs\\%s_list.txt' % subset  # 'E:/%s_list.txt' % subset  # sys.argv[1]
    gt_dir = 'G:/gddata/all'  # sys.argv[2]
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"
    val_list_txt = "E:\\Downloads\\mc_seg\\tifs\\val_list.txt"

    save_dir = 'E:/Downloads/tower_detection_v2/%s' % subset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    valid_labels_set = [3, 4]
    label_maps = {
        3: 'tower',
        4: 'insulator'
    }
    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]
    val_list = []
    if os.path.exists(val_list_txt):
        with open(val_list_txt, 'r', encoding='utf-8-sig') as fp:
            val_list = [line.strip() for line in fp.readlines()]

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    for p in [save_img_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    data_dict = {'images': [], 'categories': [], 'annotations': []}
    # train_dict = {'images': [], 'categories': [], 'annotations': []}
    # val_dict = {'images': [], 'categories': [], 'annotations': []}
    for idx, name in enumerate(["tower0", "tower1", "tower", "insulator"]):  # 1,2,3 is gan, 4 is jueyuanzi
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)
        # train_dict['categories'].append(single_cat)
        # val_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1

    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

    subsizes = [1024]
    scales = [1.0]
    gaps = [256] if 'train' in subset else [256]

    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue
        if subset == 'train' and file_prefix in val_list:
            continue
        if subset == 'val' and file_prefix not in val_list:
            continue

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

        print('loading gt ...')
        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                    valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        for si, (subsize, scale) in enumerate(zip(subsizes, scales)):

            for gap in gaps:
                offsets = compute_offsets(height=orig_height, width=orig_width, subsize=subsize, gap=gap)

                for oi, (xoffset, yoffset, sub_w, sub_h) in enumerate(offsets):  # left, up
                    # sub_width = min(orig_width, big_subsize)
                    # sub_height = min(orig_height, big_subsize)
                    # if xoffset + sub_width > orig_width:
                    #     sub_width = orig_width - xoffset
                    # if yoffset + sub_height > orig_height:
                    #     sub_height = orig_height - yoffset
                    print(oi, len(offsets), xoffset, yoffset, sub_w, sub_h)
                    save_prefix = '%d_%d_%d_%d' % (ti, si, oi, gap)

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
                    for bi in [0, 1, 2] if file_prefix not in bands_info else [2, 1, 0]:
                        band = ds.GetRasterBand(bi + 1)
                        band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_w, win_ysize=sub_h)
                        cutout.append(band_data)
                    cutout = np.stack(cutout, -1)  # RGB

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
                            if area1 >= 0.7 * area:
                                sub_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                            else:
                                invalid_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                    # if len(np.where(im1[:, :, 0] > 0)[0]) < 0.5 * np.prod(im1.shape[:2]):
                    #     continue

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

                    prob = np.random.rand()
                    if len(sub_gt_boxes) > 0:

                        single_image = {'file_name': save_prefix + '.jpg', 'id': image_id, 'width': sub_w,
                                        'height': sub_h}
                        data_dict['images'].append(single_image)
                        # if prob > 0.2:
                        #     train_dict['images'].append(single_image)
                        # else:
                        #     val_dict['images'].append(single_image)

                        cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                        save_img = True
                        for box2 in sub_gt_boxes:
                            xmin, ymin, xmax, ymax, label = box2

                            if save_img:
                                cv2.rectangle(cutout, (xmin, ymin), (xmax, ymax), color=colors[label],
                                              thickness=3)

                            xc1 = int((xmin + xmax) / 2)
                            yc1 = int((ymin + ymax) / 2)
                            w1 = xmax - xmin
                            h1 = ymax - ymin
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
                            # if prob > 0.2:
                            #     train_dict['annotations'].append(single_obj)
                            # else:
                            #     val_dict['annotations'].append(single_obj)

                            inst_count = inst_count + 1

                        image_id = image_id + 1

                        if save_img:
                            cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR
                    # else:
                    #     # if no gt_boxes
                    #     if len(np.where(cutout[:, :, 0] > 0)[0]) < 0.5 * np.prod(cutout.shape[:2]):
                    #         continue
                    #
                    #     if np.random.rand() < 0.1:
                    #         # for yolo format
                    #         save_prefix = save_prefix + '_noGT'
                    #         cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                    # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                    # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                    # ims.append(im)

    if inst_count > 1:
        with open(save_dir + '/%s.json' % subset, 'w') as f_out:
            json.dump(data_dict, f_out, indent=4)
        # with open(save_dir + '/train.json', 'w') as f_out:
        #     json.dump(train_dict, f_out, indent=4)
        # with open(save_dir + '/val.json', 'w') as f_out:
        #     json.dump(val_dict, f_out, indent=4)


def gen_tower_detection_dataset_crossvalidation():
    source = 'G:/gddata/all'  # 'E:/%s_list.txt' % subset  # sys.argv[1]
    gt_dir = 'G:/gddata/all'  # sys.argv[2]
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"

    valid_labels_set = [3, 4]
    label_maps = {
        3: 'tower',
        4: 'insulator'
    }
    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    tif_prefixes = []
    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue
        tif_prefixes.append(file_prefix)

    prefix_indices = np.array_split(np.arange(len(tif_prefixes)), indices_or_sections=5)
    for cv_i, prefix_indice in enumerate(prefix_indices):
        val_prefixes = [tif_prefixes[i] for i in prefix_indice]
        train_indice = set(np.arange(len(tif_prefixes))) - set(prefix_indice)
        train_prefixes = [tif_prefixes[i] for i in list(train_indice)]
        save_dir = 'E:/Downloads/tower_detection_crossvalidation/data_%d' % cv_i
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_img_path = '%s/images/' % save_dir
        save_img_shown_path = '%s/images_shown/' % save_dir
        for p in [save_img_path, save_img_shown_path]:
            if not os.path.exists(p):
                os.makedirs(p)

        with open(os.path.join(save_dir, 'train_prefixes.txt'), 'w') as fp:
            fp.writelines([prefix + '\n' for prefix in train_prefixes])
        with open(os.path.join(save_dir, 'val_prefixes.txt'), 'w') as fp:
            fp.writelines([prefix + '\n' for prefix in val_prefixes])

        data_dict = {'images': [], 'categories': [], 'annotations': []}
        train_dict = {'images': [], 'categories': [], 'annotations': []}
        val_dict = {'images': [], 'categories': [], 'annotations': []}
        for idx, name in enumerate(["tower0", "tower1", "tower", "insulator"]):  # 1,2,3 is gan, 4 is jueyuanzi
            single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
            data_dict['categories'].append(single_cat)
            train_dict['categories'].append(single_cat)
            val_dict['categories'].append(single_cat)

        inst_count = 1
        image_id = 1

        colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}

        subsizes = [1024]
        scales = [1.0]
        gaps = [256]

        for ti in range(len(tiffiles)):
            tiffile = tiffiles[ti]
            file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
            if 'Original' in file_prefix or file_prefix in invalid_tifs:
                continue
            if file_prefix not in val_prefixes:
                is_train = True
            else:
                is_train = False

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

            print('loading gt ...')
            gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
            gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

            gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                        valid_labels=valid_labels_set)

            if len(gt_boxes) == 0:
                continue

            for si, (subsize, scale) in enumerate(zip(subsizes, scales)):

                for gap in gaps:
                    offsets = compute_offsets(height=orig_height, width=orig_width, subsize=subsize, gap=gap)

                    for oi, (xoffset, yoffset, sub_w, sub_h) in enumerate(offsets):  # left, up
                        # sub_width = min(orig_width, big_subsize)
                        # sub_height = min(orig_height, big_subsize)
                        # if xoffset + sub_width > orig_width:
                        #     sub_width = orig_width - xoffset
                        # if yoffset + sub_height > orig_height:
                        #     sub_height = orig_height - yoffset
                        print(oi, len(offsets), xoffset, yoffset, sub_w, sub_h)
                        save_prefix = '%d_%d_%d_%d' % (ti, si, oi, gap)

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
                        for bi in [0, 1, 2] if file_prefix not in bands_info else [2, 1, 0]:
                            band = ds.GetRasterBand(bi + 1)
                            band_data = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_w, win_ysize=sub_h)
                            cutout.append(band_data)
                        cutout = np.stack(cutout, -1)  # RGB

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
                                if area1 >= 0.7 * area:
                                    sub_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])
                                else:
                                    invalid_gt_boxes.append([xmin1, ymin1, xmax1, ymax1, label1])

                        # if len(np.where(im1[:, :, 0] > 0)[0]) < 0.5 * np.prod(im1.shape[:2]):
                        #     continue

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

                        if len(sub_gt_boxes) > 0:

                            single_image = {'file_name': save_prefix + '.jpg', 'id': image_id, 'width': sub_w,
                                            'height': sub_h}
                            data_dict['images'].append(single_image)
                            if is_train:
                                train_dict['images'].append(single_image)
                            else:
                                val_dict['images'].append(single_image)

                            cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                            save_img = True
                            for box2 in sub_gt_boxes:
                                xmin, ymin, xmax, ymax, label = box2

                                if save_img:
                                    cv2.rectangle(cutout, (xmin, ymin), (xmax, ymax), color=colors[label],
                                                  thickness=3)

                                xc1 = int((xmin + xmax) / 2)
                                yc1 = int((ymin + ymax) / 2)
                                w1 = xmax - xmin
                                h1 = ymax - ymin
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
                                if is_train:
                                    train_dict['annotations'].append(single_obj)
                                else:
                                    val_dict['annotations'].append(single_obj)

                                inst_count = inst_count + 1

                            image_id = image_id + 1

                            # if save_img:
                            #     cv2.imwrite(save_img_shown_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR
                        # else:
                        #     # if no gt_boxes
                        #     if len(np.where(cutout[:, :, 0] > 0)[0]) < 0.5 * np.prod(cutout.shape[:2]):
                        #         continue
                        #
                        #     if np.random.rand() < 0.1:
                        #         # for yolo format
                        #         save_prefix = save_prefix + '_noGT'
                        #         cv2.imwrite(save_img_path + save_prefix + '.jpg', cutout[:, :, ::-1])  # RGB --> BGR

                        # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                        # im /= 255.0  # 0 - 255 to 0.0 - 1.0
                        # ims.append(im)

        if inst_count > 1:
            with open(save_dir + '/all.json', 'w') as f_out:
                json.dump(data_dict, f_out, indent=4)
            with open(save_dir + '/train.json', 'w') as f_out:
                json.dump(train_dict, f_out, indent=4)
            with open(save_dir + '/val.json', 'w') as f_out:
                json.dump(val_dict, f_out, indent=4)


def test_shp():
    reference_tif_filename = r'G:\gddata\all\2-WV03-在建杆塔.tif'
    name1 = r'E:\Downloads\mc_seg\logs\U_Net_512_4_0.0001\epoch-50\test_tif\2-WV03-在建杆塔\landslide.shp'
    name2 = r'E:\Downloads\mc_seg\logs\SMP_UnetPlusPlus_512_8_0.001\epoch-50\test_tif\2-WV03-在建杆塔\landslide.shp'
    driver = ogr.GetDriverByName("ESRI Shapefile")
    s1 = driver.Open(name1)
    s2 = driver.Open(name2)
    ds = gdal.Open(reference_tif_filename, gdal.GA_ReadOnly)
    gdal_trans_info = ds.GetGeoTransform()
    gdal_projection = ds.GetProjection()
    gdal_projection_sr = osr.SpatialReference(wkt=gdal_projection)
    ds = None
    layer1 = s1.GetLayer()
    layer2 = s2.GetLayer()
    # outArea = []

    label_name = "landslide"
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    shp_filename = r'E:\Downloads\mc_seg\logs\U_Net_512_4_0.0001\epoch-50\test_tif\2-WV03-在建杆塔\landslide_inter.shp'
    outDataSource = outDriver.CreateDataSource(shp_filename)
    outLayer = outDataSource.CreateLayer(label_name, gdal_projection_sr, geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    newField = ogr.FieldDefn("Class", ogr.OFTString)
    outLayer.CreateField(newField)

    for feat1 in layer1:
        geom1 = feat1.GetGeometryRef()
        for feat2 in layer2:
            geom2 = feat2.GetGeometryRef()
            if geom2.Intersects(geom1):
                inter = geom2.Intersection(geom1)
                print(type(geom1), type(inter))
                if inter is not None:
                    # outArea.append(inter.GetArea())

                    # add new geom to layer
                    outFeature = ogr.Feature(featureDefn)
                    outFeature.SetGeometry(inter)
                    outFeature.SetField("Class", label_name)
                    outLayer.CreateFeature(outFeature)
                    outFeature.Destroy()
        layer2.ResetReading()
    # print(outArea)
    featureDefn = None
    outLayer = None
    outDataSource.Destroy()
    outDataSource = None


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels):
    dices = np.zeros(numLabels, dtype=np.float32)
    for index in range(numLabels):
        tmp_dice = dice_coef(y_true[:,:,index], y_pred[:,:,index])
        dices[index] = tmp_dice
    return dices


def test_multiclass_dice():
    gt_images_dir = r'E:\line_foreign_object_detection\augmented_data_v2\val\annotations_with_foreign'
    pred_images_dir = r'E:\line_foreign_object_detection\logs_v2\U_Net_512_2_0.001\val'

    valid_labels = [1, 2]
    lines = []
    all_dices = []
    for gt_filename in glob.glob(os.path.join(gt_images_dir, '*.png')):
        prefix = os.path.basename(gt_filename).replace('.png', '')
        pred_filename = os.path.join(pred_images_dir, prefix + '_binary.png')
        # if True:
        #     gt_filename = r'E:\line_foreign_object_detection\augmented_data\val\annotations_with_foreign\bg_0_0_0_0000000051.png'
        #     pred_filename = r'E:\line_foreign_object_detection\logs\U_Net_512_2_0.001\val\bg_0_0_0_0000000051_binary.png'
        #     gt = cv2.imread(gt_filename)[:, :, 0]
        #     pred = cv2.imread(pred_filename)[:, :, 0]
        # else:
        #     gt = np.zeros((200, 200), dtype=np.uint8)
        #     pred = np.zeros((200, 200), dtype=np.uint8)
        #     gt[50:100, 50:100] = 1
        #     gt[110:150, 110:130] = 2
        #     pred[40:90, 40:90] = 1
        #     pred[100:140, 90:150] =2

        if os.path.exists(gt_filename) and os.path.exists(pred_filename):
            gt = cv2.imread(gt_filename)[:, :, 0]
            pred = cv2.imread(pred_filename)[:, :, 0]
            H, W = gt.shape[:2]
            gt_onehot = np.zeros((H, W, len(valid_labels)), dtype=np.float32)
            pred_onehot = np.zeros((H, W, len(valid_labels)), dtype=np.float32)
            for i, label in enumerate(valid_labels):
                gt_onehot[gt == label, i] = 1
                pred_onehot[pred == label, i] = 1

            tmp_dices = dice_coef_multilabel(gt_onehot, pred_onehot, len(valid_labels))
            if tmp_dices[0] > 0.5:
                lines.append('%s,%f,%f\n' % (prefix, tmp_dices[0], tmp_dices[1]))
                all_dices.append(tmp_dices)
    if len(lines) > 0:
        with open(os.path.join(pred_images_dir, 'results.csv'), 'w') as fp:
            fp.writelines(lines)

        all_dices = np.stack(all_dices, axis=1)
        print('all_dices')
        print(all_dices)
        print(np.mean(all_dices, axis=1))


def split_train_val_set():
    from sklearn.cluster import KMeans
    source = 'G:/gddata/all'  # 'E:/%s_list.txt' % subset  # sys.argv[1]
    gt_dir = 'G:/gddata/all'  # sys.argv[2]
    bands_info_txt = "E:\\Downloads\\mc_seg\\tifs\\bands_info.txt"
    invalid_tifs_txt = "E:\\Downloads\\mc_seg\\tifs\\invalid_tifs.txt"

    valid_labels_set = [3, 4]
    label_maps = {
        3: 'tower',
        4: 'insulator'
    }
    bands_info = []
    if os.path.exists(bands_info_txt):
        with open(bands_info_txt, 'r', encoding='utf-8-sig') as fp:
            bands_info = [line.strip() for line in fp.readlines()]
    invalid_tifs = []
    if os.path.exists(invalid_tifs_txt):
        with open(invalid_tifs_txt, 'r', encoding='utf-8-sig') as fp:
            invalid_tifs = [line.strip() for line in fp.readlines()]

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    tif_prefixes = []
    for ti in range(len(tiffiles)):
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix or file_prefix in invalid_tifs:
            continue
        tif_prefixes.append(file_prefix)

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

        print('loading gt ...')
        gt_txt_filename = os.path.join(gt_dir, file_prefix + '_gt.txt')
        gt_xml_filename = os.path.join(gt_dir, file_prefix + '_gt_5.xml')

        gt_boxes, gt_labels = load_gt_for_detection(gt_txt_filename, gt_xml_filename, gdal_trans_info=geotransform,
                                                    valid_labels=valid_labels_set)

        if len(gt_boxes) == 0:
            continue

        gt_boxes1 = np.copy(gt_boxes)
        xmin, ymin = np.min(gt_boxes1[:, 0:2], axis=0)
        xmax, ymax = np.max(gt_boxes1[:, 2:4], axis=0)
        w, h = xmax - xmin, ymax - ymin

        box_centers = []
        for box in gt_boxes1:
            x1, y1, x2, y2 = box
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2
            box_centers.append([xc, yc])
        box_centers = np.array(box_centers).reshape(-1, 2)

        if w > h:
            idx = np.argsort(box_centers[:, 0])
        else:
            idx = np.argsort(box_centers[:, 1])

        gt_boxes1 = gt_boxes1[idx]
        box_centers = box_centers[idx]
        idx = int(np.floor(0.8 * len(box_centers)))
        if idx == box_centers.shape[0] - 1:
            idx = box_centers.shape[0] - 2
        xc = (box_centers[idx, 0] + box_centers[idx + 1, 0]) // 2
        yc = (box_centers[idx, 1] + box_centers[idx + 1, 1]) // 2
        if w > h:
            pass
        else:
            pass


def rotate_image(im, boxes, degree=0):
    # degree should be 90, 180, 270
    if degree == 0:
        return im, boxes

    if degree == 90:
        im_new = np.rot90(im, k=1).copy()
    elif degree == 180:
        im_new = np.rot90(im, k=2).copy()
    elif degree == 270:
        im_new = np.rot90(im, k=3).copy()
    else:
        im_new = im.copy()

    new_boxes = []
    H, W = im.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = [box[i] for i in range(4)]
        if degree == 90:
            new_boxes.append([y1, W-x2, y2, W-x1] + box[4:])
        elif degree == 180:
            new_boxes.append([W-x2, H-y2, W-x1, H-y1] + box[4:])
        elif degree == 270:
            new_boxes.append([H-y2, x1, H-y1, x2] + box[4:])
        else:
            new_boxes.append(box)
    return im_new, new_boxes


def plot_boxes(im, boxes):
    im = im.copy()
    for box in boxes:
        x1, y1, x2, y2 = [box[i] for i in range(4)]
        cv2.rectangle(im, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)
        cv2.circle(im, center=(x1, y1), radius=3, color=(0, 0, 255), thickness=2)
    return im


def test_rotation():
    save_dir = "E:/"
    im = np.zeros((200, 100, 3), dtype=np.uint8)
    im[0:10, 0:10, :] = 255
    boxes = np.array([[10, 20, 50, 60, 1]])
    cv2.imwrite(os.path.join(save_dir, 'degree-0.png'), plot_boxes(im, boxes))

    for degree in [0, 90, 180, 270]:
        im_new, boxes_new = rotate_image(im, boxes, degree=degree)
        cv2.imwrite(os.path.join(save_dir, 'degree-%d.png' % degree), plot_boxes(im_new, boxes_new))


def test_hist_matching():
    from skimage import exposure
    images_dir = 'E:/Downloads/tower_detection/v4_augtimes5_800_200_noSplit/images'
    save_dir = 'E:/Downloads/tower_detection/test_hist_matching'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filenames = glob.glob(os.path.join(images_dir, '*.jpg'))
    indices = np.arange(len(filenames))
    for j in range(10):
        i = np.random.choice(indices, replace=False, size=2)
        filename1, filename2 = filenames[i[0]], filenames[i[1]]
        prefix1 = os.path.basename(filename1).replace('.jpg', '')
        prefix2 = os.path.basename(filename2).replace('.jpg', '')
        img1 = cv2.imread(filename1)
        img2 = cv2.imread(filename2)
        multi = True if img1.shape[-1] > 1 else False
        matched1 = exposure.match_histograms(img1, img2, multichannel=multi)
        matched2 = exposure.match_histograms(img2, img1, multichannel=multi)
        cv2.imwrite(os.path.join(save_dir, '%s_%s.jpg' % (prefix1, prefix2)),
                    np.concatenate([img1, img2, matched1, matched2], axis=1))


if __name__ == '__main__':
    action = sys.argv[1]
    print(action)
    if action == 'gen_dataset':
        main(subset='train')
        main(subset='val')
    elif action == 'gen_dataset_shp':
        main_using_shp_single_tif(subset='train')
        main_using_shp_single_tif(subset='val')
    elif action == 'gen_dataset_shp_multi_tif':
        main_using_shp_multi_tif(subset='train', random_count=200)
        main_using_shp_multi_tif(subset='val', random_count=200)
    elif action == 'extract_towers':
        method = sys.argv[2]
        res_in_cm = int(float(sys.argv[3]))
        extract_towers_using_shp_multi_tif(subset='train', resample_method=method, res_in_cm=res_in_cm)
    elif action == 'xml2shp':
        convert_envi_xml_to_separate_shapefile()
    elif action == 'towerxml2shp':
        convert_envi_tower_xml_to_seperate_shapefile()
    elif action == 'resample':
        method = sys.argv[2]
        res_in_cm = int(float(sys.argv[3]))
        resample_aerial_tifs(method=method, res_in_cm=res_in_cm)
    elif action == 'gen_tower_detection_dataset':
        gen_tower_detection_dataset()
    elif action == 'gen_tower_detection_dataset_v3':
        gen_tower_detection_dataset_v3()
    elif action == 'gen_tower_detection_dataset_v4':
        # gen_tower_detection_dataset_v4(aug_times=5, subset='train')
        # gen_tower_detection_dataset_v4(aug_times=5, subset='val')
        gen_tower_detection_dataset_v4(aug_times=5, subset=None)
    elif action == 'gen_tower_detection_dataset_v5':
        gen_tower_detection_dataset_v5(aug_times=5, subset='train')
        gen_tower_detection_dataset_v5(aug_times=5, subset='val')
        # gen_tower_detection_dataset_v5(aug_times=5, subset=None)
    elif action == 'gen_tower_detection_dataset_v2':
        gen_tower_detection_dataset_v2(subset='train')
        gen_tower_detection_dataset_v2(subset='val')
    elif action == 'gen_tower_detection_dataset_crossvalidation':
        gen_tower_detection_dataset_crossvalidation()
    elif action == 'test_shp':
        test_shp()
    elif action == 'test_multiclass_dice':
        test_multiclass_dice()
    elif action == 'split_train_val_set':
        split_train_val_set()
    elif action == 'test_rotation':
        test_rotation()
    elif action == 'test_hist_matching':
        test_hist_matching()
