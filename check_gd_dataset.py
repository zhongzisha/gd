import copy
import sys,os,glob,shutil,time

from osgeo import gdal,ogr,osr
import numpy as np
import cv2

from myutils import load_gt_for_detection


src_dir = r'G:\gddata\all'
dst_dir = r'E:\gddata_processed'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)


def get_all_metadata():

    input_filenames = glob.glob(os.path.join(src_dir, '*.tif'))

    lines = []
    for filename in input_filenames:
        file_prefix = filename.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix:
            continue

        ds = gdal.Open(filename, gdal.GA_ReadOnly)

        trans = ds.GetGeoTransform()
        proj = ds.GetProjection()

        if trans is None:
            trans = ''
        if proj is None:
            proj = ''

        lines.append('{}\n{}\n{}\n\n\n'.format(file_prefix, proj, trans))

        ds = None

    with open(os.path.join(dst_dir, 'metadata.csv'), 'w') as fp:
        fp.writelines(lines)


def add_metadata():
    maps = {
        '110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）': '110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）',
        '110kv莱金线N26-N33_N38-N39_N17-N18（杆塔、导线、绝缘子、树木）': '110kv苏程线N3-N17（杆塔、导线、绝缘子、树木）',
        '220kvchangmianxiann31-n36':'220kvqinshunxiann53-n541',
        '220kvchangmianxiann66-n68':'220kvqinshunxiann53-n541',
        '220kvchangmianxiann74-n82':'220kvqinshunxiann53-n541',
        '220kvqinshunxiann39-n42':'220kvqinshunxiann70-n71',
        '220kvqinshunxiann64-n65':'220kvqinshunxiann70-n71',
        '220kv厂梅线13-14（杆塔、导线、绝缘子、树木）':'220kvqinshunxiann70-n71',
        '220kv长顺线N51-N55_0.05m_杆塔、导线、绝缘子、树木':'220kvqinshunxiann70-n71'
    }

    driver = gdal.GetDriverByName("GTiff")

    for k, v in maps.items():
        src_filename = os.path.join(src_dir, k + '.tif')
        dst_filename = os.path.join(dst_dir, k + '.tif')
        tmp_filename = os.path.join(dst_dir, '_nocompressed.tif')
        tmp2_filename = os.path.join(dst_dir, '_BeforeReproj.tif')
        ref_fileanme = os.path.join(src_dir, v + '.tif')

        if os.path.exists(src_filename) and os.path.exists(ref_fileanme):
            src_ds = gdal.Open(src_filename, gdal.GA_ReadOnly)
            ref_ds = gdal.Open(ref_fileanme, gdal.GA_ReadOnly)

            src_trans = src_ds.GetGeoTransform()
            ref_trans = ref_ds.GetGeoTransform()
            src_proj = src_ds.GetProjection()
            ref_proj = ref_ds.GetProjection()

            if True:
                dst_ds = driver.CreateCopy(tmp_filename, src_ds, strict=0)
                dst_ds.SetGeoTransform(ref_trans)
                dst_ds.SetProjection(ref_proj)
                dst_ds.FlushCache()

                dst_ds = None
                ref_ds = None
                src_ds = None

                if os.path.exists(tmp_filename):
                    print('reproject file')
                    time.sleep(3)
                    command = r'gdalwarp -t_srs ESRI::"%s\32650.prj" %s %s' % \
                              (os.path.dirname(__file__), tmp_filename, tmp2_filename)
                    os.system(command)
                    time.sleep(3)
                    if os.path.exists(tmp_filename):
                        os.remove(tmp_filename)
                        time.sleep(1)

                if os.path.exists(tmp2_filename):
                    print('compressing the result file')
                    time.sleep(3)
                    command = 'gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=LZW" -co "BIGTIFF=YES" %s %s' % \
                              (tmp2_filename, dst_filename)
                    os.system(command)
                    time.sleep(3)
                    if os.path.exists(tmp2_filename):
                        os.remove(tmp2_filename)
                        time.sleep(1)

            # change xml filename
            src_xml_filenames = glob.glob(os.path.join(src_dir, k + '*.xml'))
            for src_xml_filename in src_xml_filenames:
                xml_postfix = src_xml_filename.split(os.sep)[-1].replace(k, '')
                ref_xml_filename = os.path.join(src_dir, v + xml_postfix)
                if (not os.path.exists(src_xml_filename)) or (not os.path.exists(ref_xml_filename)):
                    continue
                with open(src_xml_filename, 'r') as fp:
                    src_lines = fp.readlines()
                with open(ref_xml_filename, 'r') as fp:
                    ref_lines = fp.readlines()
                ref_coord_line = None
                for line in ref_lines:
                    if 'CoordSysStr' in line:
                        ref_coord_line = line
                        break
                if ref_coord_line is not None:
                    dst_lines = copy.deepcopy(src_lines)
                    for i, line in enumerate(dst_lines):
                        if 'CoordSysStr' in line:
                            dst_lines[i] = ref_coord_line
                        line1 = line.strip()
                        if line1[0] != '<':
                            coords = [float(val) for val in line1.split(' ')]
                            mapcoords = []
                            for j in range(len(coords)//2):
                                xmin = coords[2*j]
                                ymin = coords[2*j + 1]
                                x1 = ref_trans[0] + (xmin + 0.5) * ref_trans[1] + (ymin + 0.5) * \
                                     ref_trans[2]
                                y1 = ref_trans[3] + (xmin + 0.5) * ref_trans[4] + (ymin + 0.5) * \
                                     ref_trans[5]
                                mapcoords += [x1, y1]
                            dst_lines[i] = ' '.join(['%.6f' % val for val in mapcoords]) + '\n'
                    dst_xml_filename = os.path.join(dst_dir, k + xml_postfix)
                    with open(dst_xml_filename, 'w') as fp:
                        fp.writelines(dst_lines)

            print(k)

            # break

    driver = None


def extract_fg_images():
    src_dir = r'E:\gddata_processed'
    dst_dir = r'E:\gddata_processed_reprojected'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    input_filenames = glob.glob(os.path.join(src_dir, '*.tif'))

    for filename in input_filenames:
        file_prefix = filename.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix:
            continue

        print(file_prefix)

        ds = gdal.Open(filename, gdal.GA_ReadOnly)

        trans = ds.GetGeoTransform()
        proj = ds.GetProjection()

        txt_filename = os.path.join(src_dir, file_prefix + '_gt_5.txt')
        xml_filename = os.path.join(src_dir, file_prefix + '_gt_5.xml')
        if not os.path.exists(xml_filename):
            continue

        gt_boxes, gt_labels = load_gt_for_detection(txt_filename,
                                                    xml_filename,
                                                    trans,
                                                    valid_labels=[3,4])

        if len(gt_boxes) == 0:
            continue

        save_dir = os.path.join(dst_dir, file_prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            print(i, box, label)

            if int(label) == 3:
                xmin, ymin, xmax, ymax = box
                xoffset = xmin
                yoffset = ymin
                width = xmax - xmin
                height = ymax - ymin
                cutout = []
                for bi in range(3):
                    band = ds.GetRasterBand(bi + 1)
                    band_data = band.ReadAsArray(int(xoffset), int(yoffset),
                                                 win_xsize=int(width),
                                                 win_ysize=int(height))
                    cutout.append(band_data)
                cutout = np.stack(cutout, -1)  # this is RGB

                # cv2.imwrite('%s/%d.png' % (save_dir, i), cutout[:, :, ::-1])
                cv2.imencode('.png', cutout[:, :, ::-1])[-1].tofile('%s/%d.png' % (save_dir, i))

        ds = None


def convert_coordinate_system():

    input_filenames = glob.glob(os.path.join(src_dir, '*.tif'))

    lines = []
    for filename in input_filenames:
        file_prefix = filename.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix:
            continue

        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        trans = ds.GetGeoTransform()
        proj = ds.GetProjection()
        ds = None

        if trans is None:
            trans = ''
        if proj is None:
            proj = ''

        if proj[:5] != 'PROJC':
            lines.append('{}\n{}\n{}\n\n\n'.format(file_prefix, proj, trans))

            tmp_filename = os.path.join(dst_dir, file_prefix + '_tmp.tif')
            if not os.path.exists(tmp_filename):
                print('reproject file')
                time.sleep(3)
                command = r'gdalwarp -t_srs ESRI::"F:\gd\32650.prj" %s %s' % \
                          (filename, tmp_filename)
                os.system(command)
                time.sleep(3)

            dst_filename = os.path.join(dst_dir, file_prefix + '.tif')
            if os.path.exists(tmp_filename):
                print('compressing the result file')
                time.sleep(3)
                command = 'gdal_translate -of GTiff -co "TILED=YES" -co "COMPRESS=LZW" -co "BIGTIFF=YES" %s %s' % \
                          (tmp_filename, dst_filename)
                os.system(command)
                time.sleep(3)
                if os.path.exists(tmp_filename):
                    os.remove(tmp_filename)
                    time.sleep(1)
            print(file_prefix)

    with open(os.path.join(dst_dir, 'geogcs.csv'), 'w') as fp:
        fp.writelines(lines)


def remove_metadata():
    input_filenames = glob.glob(os.path.join(src_dir, '*.tif'))

    dst_dir1 = r'E:\gddata_processed_changedProjs'
    if not os.path.exists(dst_dir1):
        os.makedirs(dst_dir1)

    driver = gdal.GetDriverByName("GTiff")

    lines = []
    for filename in input_filenames:
        file_prefix = filename.split(os.sep)[-1].replace('.tif', '')
        if 'Original' in file_prefix:
            continue

        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        trans = ds.GetGeoTransform()
        proj = ds.GetProjection()

        xOrigin = trans[0]
        yOrigin = trans[3]
        pixelWidth = trans[1]
        pixelHeight = trans[5]

        if trans[2] != 0 or trans[4] != 0:  # if not north
            import pdb
            pdb.set_trace()

        if proj[:5] != 'PROJC':
            lines.append('{}\n{}\n{}\n\n\n'.format(file_prefix, proj, trans))

            tmp_filename = os.path.join(dst_dir1, file_prefix + '_tmp.tif')
            if not os.path.exists(tmp_filename):
                print('reproject file')
                time.sleep(3)
                command = r'gdalwarp -t_srs ESRI::"F:\gd\32650.prj" %s %s' % \
                          (filename, tmp_filename)
                os.system(command)
                time.sleep(3)

            ds2 = gdal.Open(tmp_filename, gdal.GA_ReadOnly)
            ref_trans = ds2.GetGeoTransform()
            ref_proj = ds2.GetProjection()
            ds2 = None

            time.sleep(3)
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
                time.sleep(1)

            ds2 = driver.CreateCopy(os.path.join(dst_dir1, file_prefix + '.tif'), ds, strict=0)
            ds2.SetGeoTransform(ref_trans)
            ds2.SetProjection(ref_proj)
            ds2.FlushCache()
            ds2 = None
            time.sleep(4)

            # change xml filename
            src_xml_filenames = glob.glob(os.path.join(src_dir, file_prefix + '*.xml'))
            for src_xml_filename in src_xml_filenames:
                xml_postfix = src_xml_filename.split(os.sep)[-1].replace(file_prefix, '')
                with open(src_xml_filename, 'r') as fp:
                    src_lines = fp.readlines()

                ref_coord_line = '      <CoordSysStr>%s</CoordSysStr>\n' % ref_proj
                if ref_coord_line is not None:
                    dst_lines = copy.deepcopy(src_lines)
                    for i, line in enumerate(dst_lines):
                        if 'CoordSysStr' in line:
                            dst_lines[i] = ref_coord_line
                        line1 = line.strip()
                        if line1[0] != '<':
                            mapcoords = [float(val) for val in line1.split(' ')]
                            newmapcoords = []

                            for j in range(len(mapcoords) // 2):
                                xmin = mapcoords[2 * j]
                                ymin = mapcoords[2 * j + 1]

                                xmin1 = (xmin - xOrigin) / pixelWidth + 0.5
                                ymin1 = (ymin - yOrigin) / pixelHeight + 0.5

                                x1 = ref_trans[0] + (xmin1 + 0.5) * ref_trans[1] + (ymin1 + 0.5) * \
                                     ref_trans[2]
                                y1 = ref_trans[3] + (xmin1 + 0.5) * ref_trans[4] + (ymin1 + 0.5) * \
                                     ref_trans[5]

                                newmapcoords += [x1, y1]
                            dst_lines[i] = ' '.join(['%.6f' % val for val in newmapcoords]) + '\n'
                    dst_xml_filename = os.path.join(dst_dir1, file_prefix + xml_postfix)
                    with open(dst_xml_filename, 'w') as fp:
                        fp.writelines(dst_lines)

            # break

        ds = None


if __name__ == '__main__':
    # get_all_metadata()
    # add_metadata()
    # extract_fg_images()

    # convert_coordinate_system()

    remove_metadata()



















