import copy
import sys,os,glob,shutil,time

from osgeo import gdal,ogr,osr

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

        lines.append('{},{},{}\n'.format(file_prefix, proj, trans))

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

            if False:
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
                    command = r'gdalwarp -t_srs ESRI::"E:\test_resampling\32650.prj" %s %s' % \
                              (tmp_filename, tmp2_filename)
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


if __name__ == '__main__':
    # get_all_metadata()
    add_metadata()




















