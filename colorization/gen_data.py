import sys,os,glob,shutil,time
from osgeo import gdal,ogr,osr

import numpy as np
import cv2

from myutils import load_gt_for_detection, compute_offsets


src_dir = r'G:\gddata\all'
dst_dir = r'E:\gd_colorization'


if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)


def extract_images():

    # input_filenames = glob.glob(os.path.join(src_dir, '*.tif'))
    input_filenames = [
        "G:\\gddata\\all\\威华300m_mosaic.tif",
        "G:\\gddata\\all\\工业园350m_mosaic.tif",
        "G:\\gddata\\all\\110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）.tif",
        "G:\\gddata\\all\\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif",
        "G:\\gddata\\all\\110kv苏程线N3-N17（杆塔、导线、绝缘子、树木）.tif",
        "G:\\gddata\\all\\110kv莱金线N26-N33_N38-N39_N17-N18（杆塔、导线、绝缘子、树木）.tif",
        "G:\\gddata\\all\\220kvchangmianxiann31-n36.tif",
        "G:\\gddata\\all\\220kvqinshunxiann64-n65.tif",
        "G:\\gddata\\all\\220kvchangmianxiann74-n82.tif",
        "G:\\gddata\\all\\220kvqinshunxiann70-n71.tif",
        "G:\\gddata\\all\\220kv厂梅线13-14（杆塔、导线、绝缘子、树木）.tif",
        "G:\\gddata\\all\\220kv长顺线N51-N55_0.05m_杆塔、导线、绝缘子、树木.tif",
        "G:\\gddata\\all\\候村250m_mosaic.tif",
        "G:\\gddata\\all\\水口300m_mosaic.tif",
        "G:\\gddata\\all\\220kvqinshunxiann53-n541.tif",
        "G:\\gddata\\all\\110kv苏隆线N3-N10（杆塔、导线、绝缘子、树木）.tif",
    ]

    for fi, filename in enumerate(input_filenames):
        file_prefix = filename.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix:
            continue

        print(file_prefix)

        ds = gdal.Open(filename, gdal.GA_ReadOnly)

        projection = ds.GetProjection()
        projection_sr = osr.SpatialReference(wkt=projection)
        projection_esri = projection_sr.ExportToWkt(["FORMAT=WKT1_ESRI"])
        geotransform = ds.GetGeoTransform()
        xOrigin = geotransform[0]
        yOrigin = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        orig_height, orig_width = ds.RasterYSize, ds.RasterXSize

        offsets = compute_offsets(height=orig_height, width=orig_width, subsize=800, gap=8)

        for oi, (xoffset, yoffset, sub_w, sub_h) in enumerate(offsets):  # left, up
            # sub_width = min(orig_width, big_subsize)
            # sub_height = min(orig_height, big_subsize)
            # if xoffset + sub_width > orig_width:
            #     sub_width = orig_width - xoffset
            # if yoffset + sub_height > orig_height:
            #     sub_height = orig_height - yoffset
            print(oi, len(offsets), xoffset, yoffset, sub_w, sub_h)

            xoffset = max(1, xoffset)
            yoffset = max(1, yoffset)
            if xoffset + sub_w > orig_width - 1:
                sub_w = orig_width - 1 - xoffset
            if yoffset + sub_h > orig_height - 1:
                sub_h = orig_height - 1 - yoffset
            xoffset, yoffset, sub_w, sub_h = [int(val) for val in
                                              [xoffset, yoffset, sub_w, sub_h]]

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
            if len(np.where(cutout[:, :, 0] > 0)[0]) < 0.8 * np.prod(cutout.shape[:2]):
                continue

            save_filename = '%s/%06d_%05d.png' % (dst_dir, fi, oi)
            cv2.imwrite(save_filename, cutout[:, :, ::-1])
            # cv2.imencode('.png', cutout[:, :, ::-1])[-1].tofile(save_filename)

        ds = None


if __name__ == '__main__':
    extract_images()

