import shutil
import sys, os, glob
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
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import joblib


LABELS_TO_NAMES = {
    3: 'tower',
    4: 'insulator',
    5: 'line_box',
    6: 'water',
    7: 'building',
    8: 'tree',
    9: 'road',
    10: 'landslide',
    14: 'line_region'
}

NAMES_TO_LABELS = {v: k for k, v in LABELS_TO_NAMES.items()}


def load_polys_from_shpfile(tif_filename, shp_filename, valid_labels):

    ds = gdal.Open(tif_filename, gdal.GA_ReadOnly)

    gdal_trans_info = ds.GetGeoTransform()
    projection = ds.GetProjection()

    xOrigin = gdal_trans_info[0]
    yOrigin = gdal_trans_info[3]
    pixelWidth = gdal_trans_info[1]
    pixelHeight = gdal_trans_info[5]

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

    polys = []
    for i, feat in enumerate(layer):
        geom = feat.GetGeometryRef()
        label = NAMES_TO_LABELS[feat.GetField(0)]
        if label in valid_labels:
            wkt_str = geom.ExportToWkt()
            start = wkt_str.find("((") + 2
            end = wkt_str.find("))")
            points = []
            for point in wkt_str[start:end].split(','):
                x, y = point.split(' ')
                x = (float(x) - xOrigin) / pixelWidth + 0.5
                y = (float(y) - yOrigin) / pixelHeight + 0.5
                points += [x, y]
            polys.append(np.array(points).reshape(-1, 2).astype(np.int32))
    shp_ds = None
    ds = None

    return polys


def split_data_for_landslide_segmentation(save_root):
    hostname = socket.gethostname()
    source = 'E:/all_tif_files.txt'

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    # tiffiles = [r'G:\gddata\all\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif']

    for tiffile in tiffiles:
        process_one_tif_split_data_for_landslide_segmentation(save_root, tiffile)


def process_one_tif_split_data_for_landslide_segmentation(save_root=None, tiffile=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        gt_dir = r'E:\gddata_processed'  # sys.argv[2]

    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
    print(file_prefix)

    model_savefilename = os.path.join(save_root, file_prefix, 'OneClassSVM.joblib')
    if not os.path.exists(model_savefilename):
        return

    # valid_labels_set = [1, 2, 3, 4]
    valid_labels_set = [10]
    palette = np.random.randint(0, 255, size=(len(valid_labels_set), 3))  # building, water, road, landslide
    palette = np.array([[255, 0, 0], [250, 0, 0]])
    opacity = 0.2

    save_dir = '%s/%s/splits' % (save_root, file_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    subsize = 2048
    gap = 16
    filename = tiffile.split(os.sep)[-1]
    # shutil.move(tif_file, save_filename)
    cmd_line2 = r"python gdal_retile.py -ps %d %d -overlap %d -targetDir %s %s" % \
                (subsize, subsize, gap, save_dir, tiffile)
    os.system(cmd_line2)
    # shutil.move(save_filename, tif_file)


def merge_data_for_landslide_segmentation(save_root):
    hostname = socket.gethostname()
    source = 'E:/all_tif_files.txt'

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    tiffiles = [r'G:\gddata\all\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif']

    for tiffile in tiffiles:
        process_one_tif_merge_data_for_landslide_segmentation(save_root, tiffile)


def process_one_tif_merge_data_for_landslide_segmentation(save_root=None, tiffile=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        gt_dir = r'E:\gddata_processed'  # sys.argv[2]

    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
    print(file_prefix)

    model_savefilename = os.path.join(save_root, file_prefix, 'OneClassSVM.joblib')
    if os.path.exists(model_savefilename):
        return

    # valid_labels_set = [1, 2, 3, 4]
    valid_labels_set = [10]
    palette = np.random.randint(0, 255, size=(len(valid_labels_set), 3))  # building, water, road, landslide
    palette = np.array([[255, 0, 0], [250, 0, 0]])
    opacity = 0.2

    save_dir = '%s/%s/merged' % (save_root, file_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    subsize = 2048
    gap = 16

    # TODO need to collect the files to merge
    names = []

    merged_filename = os.path.join(save_dir, 'merged.tif')

    if os.path.exists(merged_filename):
        os.remove(merged_filename)
    command = "gdal_merge.py -o %s -of gtiff " % (merged_filename) + " ".join(names)
    print(command)
    os.system(command)

    os.system('sleep 10')


def prepare_landslide_segmentation_dataset(save_root):
    hostname = socket.gethostname()
    source = 'E:/all_tif_files.txt'

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    # tiffiles = [r'G:\gddata\all\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif']

    for tiffile in tiffiles:
        process_one_tif(save_root, tiffile)


def check_positive_samples(kernel='linear', gamma=0.1, nu=0.5):
    from sklearn.svm import OneClassSVM
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns
    filename = r'E:\gd_newAug1_Rot0_4classes_landslide_segmentation_test\landslide_segmentation\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）\landslide_positive_samples.png'
    im = np.array(Image.open(filename))
    X = im.reshape(-1, 3)
    print(X.shape)
    X = np.unique(X, axis=0)  # remove duplicates
    print('after remove duplicates ', X.shape)
    if X.shape[0] > 100000:
        rnd_inds = np.random.choice(np.arange(X.shape[0]), replace=False, size=100000)
        X = X[rnd_inds, :]
    print(X.shape)
    svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    print(svm)

    svm.fit(X)
    pred = svm.predict(X)

    anom_index = np.where(pred == -1)
    neg_values = X[anom_index]
    print(len(neg_values))

    h = int(np.floor(np.sqrt(len(neg_values))))
    neg_im = neg_values[:(h*h), :].reshape(h, h, 3)
    plt.subplot(121)
    plt.imshow(neg_im)

    pos_index = np.where(pred == 1)
    pos_values = X[pos_index]
    h = int(np.floor(np.sqrt(len(pos_values))))
    pos_im = pos_values[:(h*h), :].reshape(h, h, 3)
    plt.subplot(122)
    plt.imshow(pos_im)
    plt.savefig(r'E:\oneclassSVM_kernel_%s_gamma=%f_nu=%f.png' % (kernel, gamma, nu))


    plt.show()

    # import pdb
    # pdb.set_trace()

    pass


def train_oneclasssvm(save_root, kernel='linear', gamma=0.1, nu=0.5):
    hostname = socket.gethostname()
    source = 'E:/all_tif_files.txt'

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    # tiffiles = [r'G:\gddata\all\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif']

    for tiffile in tiffiles:
        train_oneclasssvm_one_file(save_root, tiffile, kernel, gamma, nu)


def train_oneclasssvm_one_file(save_root, tiffile, kernel='linear', gamma=0.1, nu=0.5):

    hostname = socket.gethostname()
    if hostname == 'master':
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        gt_dir = r'E:\gddata_processed'  # sys.argv[2]

    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
    print(file_prefix)
    model_savefilename = os.path.join(save_root, file_prefix, 'OneClassSVM.joblib')
    if os.path.exists(model_savefilename):
        return

    filename = os.path.join(save_root, file_prefix, 'landslide_positive_samples.png')
    print(filename)
    if not os.path.exists(filename):
        return

    im = np.array(Image.open(filename))
    X = im.reshape(-1, 3)
    print(X.shape)
    X = np.unique(X, axis=0)  # remove duplicates
    print('after remove duplicates ', X.shape)
    if X.shape[0] > 100000:
        rnd_inds = np.random.choice(np.arange(X.shape[0]), replace=False, size=100000)
        X = X[rnd_inds, :]
    print(X.shape)
    clf = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    print(clf)
    clf.fit(X)
    pred = clf.predict(X)

    joblib.dump(clf, model_savefilename)

    anom_index = np.where(pred == -1)
    neg_values = X[anom_index]
    print(len(neg_values))

    h = int(np.floor(np.sqrt(len(neg_values))))
    neg_im = neg_values[:(h*h), :].reshape(h, h, 3)
    plt.subplot(121)
    plt.imshow(neg_im)

    pos_index = np.where(pred == 1)
    pos_values = X[pos_index]
    h = int(np.floor(np.sqrt(len(pos_values))))
    pos_im = pos_values[:(h*h), :].reshape(h, h, 3)
    plt.subplot(122)
    plt.imshow(pos_im)

    save_filename = os.path.join(save_root, file_prefix,
                                 'OneClassSVM_fit_results_kernel_%s_gamma=%f_nu=%f.png'% (kernel, gamma, nu))
    plt.savefig(save_filename)

    # plt.show()

    # import pdb
    # pdb.set_trace()


def predict_oneclasssvm(save_root, kernel='linear', gamma=0.1, nu=0.5):
    hostname = socket.gethostname()
    source = 'E:/all_tif_files.txt'

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    # tiffiles = [r'G:\gddata\all\110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif']

    for tiffile in tiffiles:
        predict_oneclasssvm_one_file(save_root, tiffile, kernel, gamma, nu)


def predict_one_sub_tifile(param):
    clf, sub_tiffilename, save_filename = param
    ds = gdal.Open(sub_tiffilename, gdal.GA_ReadOnly)
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

    # print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
    img = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)  # RGB format
    for b in range(3):
        band = ds.GetRasterBand(b + 1)
        img[:, :, b] = band.ReadAsArray(0, 0, win_xsize=orig_width, win_ysize=orig_height)
    ds = None

    img_sum = np.sum(img, axis=2)
    indices_y, indices_x = np.where(img_sum > 0)
    if len(indices_x) == 0:
        return

    pred = clf.predict(img.reshape(orig_height * orig_width, 3))
    pred[pred == -1] = 0
    pred = pred.reshape(orig_height, orig_width)

    color_seg = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
    color_seg[pred == 1, :] = np.array([250, 0, 0])

    Image.fromarray(color_seg).save(save_filename)


def predict_oneclasssvm_one_file(save_root, tiffile, kernel='linear', gamma=0.1, nu=0.5):

    hostname = socket.gethostname()
    if hostname == 'master':
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        gt_dir = r'E:\gddata_processed'  # sys.argv[2]

    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
    print(file_prefix)

    model_savefilename = os.path.join(save_root, file_prefix, 'OneClassSVM.joblib')
    if not os.path.exists(model_savefilename):
        return

    result_dir = os.path.join(save_root, file_prefix, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        return

    clf = joblib.load(model_savefilename)

    filenames = glob.glob(os.path.join(save_root, file_prefix, 'splits', '*.tif'))

    params = [
        (clf, filename, os.path.join(result_dir, filename.split(os.sep)[-1].replace('.tif', '.png')))
        for filename in filenames
    ]

    if False:
        for filename in filenames:
            save_filename = os.path.join(result_dir, filename.split(os.sep)[-1].replace('.tif', '.png'))
            print(save_filename)
            predict_one_sub_tifile(clf, sub_tiffilename=filename, save_filename=save_filename)
            print(filename, ' done')
    else:
        import multiprocessing
        from multiprocessing import Pool

        processes = 2  # multiprocessing.cpu_count()
        with Pool(processes=processes) as p:
            p.map(predict_one_sub_tifile, params)


def plot_pos_samples(pos_samples):
    from sklearn.svm import OneClassSVM
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns

    np.random.seed(13)
    X = pos_samples  # n x 3

    # np.random.seed(13)
    # x, _ = make_blobs(n_samples=200, centers=1, cluster_std=.3, center_box=(8, 8))
    #
    # plt.scatter(x[:, 0], x[:, 1])
    # plt.show()

    feat_cols = ['r', 'g', 'b']  # ['pixel' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = 1
    df['label'] = df['y'].apply(lambda i: str(i))

    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(df[feat_cols].values)
    # df['pca-one'] = pca_result[:, 0]
    # df['pca-two'] = pca_result[:, 1]
    # df['pca-three'] = pca_result[:, 2]
    # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="pca-one", y="pca-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 10),
    #     data=df.loc[rndperm, :],
    #     legend="full",
    #     alpha=0.3
    # )

    if True:
        ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
        ax.scatter(
            xs=df.loc[rndperm, :]["r"],
            ys=df.loc[rndperm, :]["g"],
            zs=df.loc[rndperm, :]["b"],
            c=df.loc[rndperm, :]["y"],
            cmap='tab10'
        )
        ax.set_xlabel('r')
        ax.set_ylabel('g')
        ax.set_zlabel('b')
        plt.show()


    if False:
        N = 10000
        df_subset = df.loc[rndperm[:N], :].copy()
        data_subset = df_subset[feat_cols].values
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data_subset)
        df_subset['pca-one'] = pca_result[:, 0]
        df_subset['pca-two'] = pca_result[:, 1]
        df_subset['pca-three'] = pca_result[:, 2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df_subset,
            legend="full",
            alpha=0.3
        )

        plt.figure(figsize=(16, 7))
        ax1 = plt.subplot(1, 2, 1)
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df_subset,
            legend="full",
            alpha=0.3,
            ax=ax1
        )
        ax2 = plt.subplot(1, 2, 2)
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df_subset,
            legend="full",
            alpha=0.3,
            ax=ax2
        )

        # pca_50 = PCA(n_components=50)
        # pca_result_50 = pca_50.fit_transform(data_subset)
        # print('Cumulative explained variation for 50 principal components: {}'.format(
        #     np.sum(pca_50.explained_variance_ratio_)))

    return

    svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
    print(svm)

    svm.fit(X)
    pred = svm.predict(X)

    anom_index = np.where(pred == -1)
    values = X[anom_index]

    # plt.scatter(x[:, 0], x[:, 1])
    # plt.scatter(values[:, 0], values[:, 1], color='r')
    # plt.show()


# random points according to the ground truth polygons
def process_one_tif(save_root=None, tiffile=None):
    hostname = socket.gethostname()
    if hostname == 'master':
        gt_dir = '/media/ubuntu/Working/rs/guangdong_aerial/all'
    else:
        gt_dir = r'E:\gddata_processed'  # sys.argv[2]

    file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
    print(file_prefix)

    # valid_labels_set = [1, 2, 3, 4]
    valid_labels_set = [10]
    palette = np.random.randint(0, 255, size=(len(valid_labels_set), 3))  # building, water, road, landslide
    palette = np.array([[255, 0, 0], [250, 0, 0]])
    opacity = 0.2

    save_dir = '%s/%s/' % (save_root, file_prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_img_path = '%s/images/' % save_dir
    save_img_shown_path = '%s/images_shown/' % save_dir
    save_lbl_path = '%s/labels/' % save_dir
    for p in [save_img_path, save_lbl_path, save_img_shown_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    for label_ind, label in enumerate(valid_labels_set):
        label_name = LABELS_TO_NAMES[label]

        if os.path.exists('%s/%s_positive_samples.png' % (save_dir, label_name)):
            check_positive_samples('%s/%s_positive_samples.png' % (save_dir, label_name))

        shp_filename = os.path.join(gt_dir, file_prefix + '_gt_%s.shp' % (label_name))

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

        if not os.path.exists(shp_filename):
            continue

        gt_polys = load_polys_from_shpfile(tiffile, shp_filename, valid_labels=[label])
        # print(gt_polys)
        if len(gt_polys) == 0:
            continue

        # 首先根据标注生成mask图像，存在内存问题！！！
        print('generate mask ...')
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        # 下面的可以直接画所有的轮廓，但是会出现相排斥的现象，用下面的循环可以得到合适的mask
        # cv2.drawContours(mask, gt_polys, -1, color=(255, 0, 0), thickness=-1)
        for poly in gt_polys:  # poly为nx2的点, numpy.array
            print(poly)
            print(len(poly))
            cv2.drawContours(mask, [poly], -1, color=(label, label, label), thickness=-1)
        time.sleep(5)

        pos_samples = []

        offsets = compute_offsets(height=orig_height, width=orig_width, subsize=1024, gap=0)
        for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up
            # sub_width = min(orig_width, big_subsize)
            # sub_height = min(orig_height, big_subsize)
            # if xoffset + sub_width > orig_width:
            #     sub_width = orig_width - xoffset
            # if yoffset + sub_height > orig_height:
            #     sub_height = orig_height - yoffset
            print(oi, len(offsets), xoffset, yoffset, sub_width, sub_height)

            xoffset = max(1, xoffset)
            yoffset = max(1, yoffset)
            if xoffset + sub_width > orig_width - 1:
                sub_width = orig_width - 1 - xoffset
            if yoffset + sub_height > orig_height - 1:
                sub_height = orig_height - 1 - yoffset
            xoffset, yoffset, sub_width, sub_height = [int(val) for val in
                                                       [xoffset, yoffset, sub_width, sub_height]]

            # sample points from mask
            seg = mask[(yoffset):(yoffset + sub_height), (xoffset):(xoffset + sub_width)]
            seg1_count = len(np.where(seg > 0)[0])
            if seg1_count < 5:
                continue

            # print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
            img = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)  # RGB format
            for b in range(3):
                band = ds.GetRasterBand(b + 1)
                img[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=sub_width, win_ysize=sub_height)

            img_sum = np.sum(img, axis=2)
            indices_y, indices_x = np.where(img_sum > 0)
            if len(indices_x) == 0:
                continue

            assert img.shape[:2] == seg.shape[:2]

            img1 = img.copy()
            if len(np.where(img1[:, :, 0] == 0)[0]) > 0.4 * np.prod(img1.shape[:2]) \
                    or len(np.where(img1[:, :, 0] == 255)[0]) > 0.4 * np.prod(img1.shape[:2]):
                continue

            minsize = min(img1.shape[:2])
            maxsize = max(img1.shape[:2])
            if maxsize > 1.5 * minsize:
                continue

            save_prefix = '%s_%d_%d' % (file_prefix, int(xoffset), int(yoffset))
            # cv2.imwrite('%s/%s.jpg' % (save_img_path, save_prefix), img)  # 不能有中文
            # cv2.imwrite('%s/%s.png' % (save_lbl_path, save_prefix), seg)
            Image.fromarray(img).save('%s/%s.jpg' % (save_img_path, save_prefix))
            Image.fromarray(seg).save('%s/%s.png' % (save_lbl_path, save_prefix))

            samples = img[np.where(seg == label)]
            if len(samples):
                pos_samples.append(samples)

            if True:
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                color_seg[seg == label, :] = palette[label_ind]

                img1 = img1 * (1 - opacity) + color_seg * opacity
                img1 = img1.astype(np.uint8)
                # cv2.imwrite('%s/%s.jpg' % (save_img_shown_path, save_prefix), img1)
                Image.fromarray(img1).save('%s/%s.jpg' % (save_img_shown_path, save_prefix))

        pos_samples = np.concatenate(pos_samples)
        if len(pos_samples):
            # import pdb
            # pdb.set_trace()
            h = int(np.floor(np.sqrt(len(pos_samples))))
            xx = pos_samples[:(h*h), :].reshape(h, h, 3)
            Image.fromarray(xx).save('%s/%s_positive_samples.png' % (save_dir, label_name))
            # plot_pos_samples(pos_samples)


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
    parser.add_argument('--debug', default=False, action='store_true')

    parser.add_argument('--oneclassSVM_gamma', default=0.1, type=float)
    parser.add_argument('--oneclassSVM_nu', default=0.1, type=float)
    parser.add_argument('--oneclassSVM_kernel', default='linear', type=str)

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
    debug = args.debug

    oneclassSVM_kernel = args.oneclassSVM_kernel
    oneclassSVM_gamma = args.oneclassSVM_gamma
    oneclassSVM_nu = args.oneclassSVM_nu

    # for detection aug
    if hostname == 'master':
        save_root = '/media/ubuntu/Data/gd_newAug%d_Rot%d_4classes_landslide_segmentation' % (aug_times, do_rotate)
    else:
        save_root = r'E:\gd_newAug%d_Rot%d_4classes_landslide_segmentation_test' % (aug_times, do_rotate)

    if aug_type == 'landslide_segmentation':
        save_root = '%s/%s' % (save_root, aug_type)
        prepare_landslide_segmentation_dataset(save_root=save_root)
        sys.exit(-1)
    elif aug_type == 'split_data_for_landslide_segmentation':
        save_root = '%s/%s' % (save_root, 'landslide_segmentation')
        split_data_for_landslide_segmentation(save_root=save_root)
    elif aug_type == 'merge_data_for_landslide_segmentation':
        save_root = '%s/%s' % (save_root, 'landslide_segmentation')
        merge_data_for_landslide_segmentation(save_root=save_root)
    elif aug_type == 'check_positive_samples':
        save_root = '%s/%s' % (save_root, 'landslide_segmentation')
        check_positive_samples(oneclassSVM_kernel, oneclassSVM_gamma, oneclassSVM_nu)
    elif aug_type == 'train_oneclasssvm':
        save_root = '%s/%s' % (save_root, 'landslide_segmentation')
        train_oneclasssvm(save_root, oneclassSVM_kernel, oneclassSVM_gamma, oneclassSVM_nu)
    elif aug_type == 'predict_oneclasssvm':
        save_root = '%s/%s' % (save_root, 'landslide_segmentation')
        predict_oneclasssvm(save_root, oneclassSVM_kernel, oneclassSVM_gamma, oneclassSVM_nu)
    else:
        pass
