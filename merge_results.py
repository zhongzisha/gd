import sys, os, glob
import cv2


def merge_using_gdal():
    # not in anaconda environment
    # in system python3 environment
    # gdal在anaconda没装好，用系统的gdal，这个程序需要在系统目录执行
    # python3 merge_results.py 0 /media/ubuntu/Temp/gd/data/aerial2/0/ /media/ubuntu/Data/gd/yoloV5/runs/detect/exp3/
    from osgeo import gdal
    import shutil
    data_id = int(sys.argv[1])
    merged_filename = 'merged_%d.tif' % data_id

    if os.path.exists(merged_filename):
        print('%s is existed.' % merged_filename)
        input()

    orig_data_root = sys.argv[2]
    result_data_root = sys.argv[3]
    save_root = './results/'
    print('orig_data_root', orig_data_root)
    tif_files = glob.glob(orig_data_root + '/*.tif')
    print(tif_files)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    names = []
    for i, tif_file in enumerate(tif_files):
        filename = tif_file.split(os.sep)[-1]
        new_filename = os.path.join(save_root, filename)
        print(filename)
        orig_filename = tif_file
        result_filename = os.path.join(result_data_root, filename)

        im = cv2.imread(result_filename)

        rows, cols, num_channels = im.shape
        cv2.putText(im, filename.replace('.tif', ''), (rows // 2, cols // 2),
                    fontFace=1, fontScale=2, color=(255, 255, 255), thickness=3)
        xmin, ymin = int(cols // 2 - 2444), int(rows // 2 - 2444)
        xmax, ymax = int(cols // 2 + 2444), int(rows // 2 + 2444)
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color=(255, 255, 255), thickness=2)

        ds = gdal.Open(orig_filename)

        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(new_filename, rows, cols, num_channels, gdal.GDT_Byte)
        outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
        for b in range(num_channels):
            band = ds.GetRasterBand(b + 1)
            arr = band.ReadAsArray()
            col, row = arr.shape[:2]
            print(b, rows, cols, row, col)
            outdata.GetRasterBand(b + 1).WriteArray(im[:row, :col, b])
            outdata.GetRasterBand(b + 1).SetNoDataValue(10000)  ##if you want these values transparent
        outdata.FlushCache()  ##saves to disk!!

        names.append(new_filename)

    if len(names) > 0:

        if os.path.exists(merged_filename):
            os.remove(merged_filename)
        command = "gdal_merge.py -o %s -of gtiff " % (merged_filename) + " ".join(names)
        print(command)
        os.system(command)

        os.system('sleep 10')

        if os.path.exists(merged_filename):
            # delete tmp directory
            # command = "rm -rf ./results/"
            # os.system(command)
            shutil.rmtree("./results/", ignore_errors=True)

            # delete the result directory
            # command = "rm -rf {}".format(result_data_root)
            # os.system(command)
            shutil.rmtree(result_data_root, ignore_errors=True)


if __name__ == '__main__':
    merge_using_gdal()
