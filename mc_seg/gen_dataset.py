import sys, os, glob, shutil

sys.path.insert(0, 'F:/gd')
import argparse
import cv2
import numpy as np
import torch
from osgeo import gdal, osr, ogr
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


def convert_envi_xml_to_separate_shapefile():
    src_dir = r"G:\gddata\all"
    xml_dir = r"E:\Downloads\mc_seg\xmls"
    shp_dir = r"E:\Downloads\mc_seg\shps"
    if not os.path.exists(shp_dir):
        os.makedirs(shp_dir)

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

        # change xml filename
        for xml_postfix in label_postfixes:
            xml_filename = os.path.join(xml_dir, file_prefix + xml_postfix + '.xml')
            if not os.path.exists(xml_filename):
                continue
            print(xml_filename)
            DomTree = xml.dom.minidom.parse(xml_filename)
            annotation = DomTree.documentElement
            regionlist = annotation.getElementsByTagName('Region')
            polys = []
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

                    shp_filename = os.path.join(shp_dir, file_prefix + '_gt_' + label_name + '.shp')
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


if __name__ == '__main__':
    action = sys.argv[1]
    print(action)
    if action == 'gen_dataset':
        main(subset='train')
        main(subset='val')
    elif action == 'xml2shp':
        convert_envi_xml_to_separate_shapefile()
