import sys, os
import numpy as np
from osgeo import gdal, osr
from natsort import natsorted
import glob
import xml.dom.minidom
from myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, \
    save_predictions_to_envi_xml
import socket


if __name__ == '__main__':

    hostname = socket.gethostname()
    if hostname == 'master':
        image_root = '/media/ubuntu/Working/rs/guangdong_aerial/'
    else:
        image_root = 'E:/gddata/'

    all_list = [
        'aerial/110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）.tif',
        'aerial/110kv江桂线N41-N42（含杆塔、导线、绝缘子、树木）.tif',
        'aerial/110kv苏程线N3-N17（杆塔、导线、绝缘子、树木）.tif',
        'aerial/110kv苏隆线N3-N10（杆塔、导线、绝缘子、树木）.tif',
        'aerial/110kv莱金线N26-N33_N38-N39_N17-N18（杆塔、导线、绝缘子、树木）.tif',
        'aerial/220kvchangmianxiann31-n36.tif',
        'aerial/220kvchangmianxiann66-n68.tif',
        'aerial/220kvchangmianxiann74-n82.tif',
        'aerial/220kvqinshunxiann39-n42.tif',
        'aerial/220kvqinshunxiann53-n541.tif',
        'aerial/220kvqinshunxiann64-n65.tif',
        'aerial/220kvqinshunxiann70-n71.tif',
        'aerial/220kv厂梅线13-14（杆塔、导线、绝缘子、树木）.tif',
        'aerial/220kv长顺线N51-N55_0.05m_杆塔、导线、绝缘子、树木.tif',
        'aerial/云南玉溪（杆塔、导线、树木、水体）.tif',
        'aerial2/候村250m_mosaic.tif',
        'aerial2/威华300m_mosaic.tif',
        'aerial2/工业园350m_mosaic.tif',
        'aerial2/水口300m_mosaic.tif'
    ]

    gt_dir = '/media/ubuntu/Data/gd_gt/'
    new_gt_dir = '/media/ubuntu/Data/gd_gt_combined/'
    if not os.path.exists(new_gt_dir):
        os.makedirs(new_gt_dir)
    gt_gap = 128

    # tiffiles = natsorted(glob.glob(orig_img_path + '/*.tif'))
    # print(tiffiles)
    tiffiles = natsorted(all_list)

    count = 0
    for i in range(len(tiffiles)):
        tiffile = os.path.join(image_root, tiffiles[i])
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        # if file_prefix != '110kv南连甲乙线N45-N50_0.05m（杆塔、导线、绝缘子、树木）':
        #     continue

        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        orig_height = ds.RasterYSize
        orig_width = ds.RasterXSize
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        projection_sr = osr.SpatialReference(wkt=projection)
        projection_esri = projection_sr.ExportToWkt(["FORMAT=WKT1_ESRI"])

        # 加载gt，分两部分，一部分是txt格式的。一部分是esri xml格式的
        gt_boxes1, gt_labels1 = load_gt_from_txt(os.path.join(gt_dir, file_prefix + '_gt.txt'))
        gt_boxes2, gt_labels2 = load_gt_from_esri_xml(os.path.join(gt_dir, file_prefix + '_gt_new.xml'),
                                                                    gdal_trans_info=geotransform)
        gt_boxes = gt_boxes1 + gt_boxes2
        gt_labels = gt_labels1 + gt_labels2
        all_boxes = np.concatenate([np.array(gt_boxes, dtype=np.float32).reshape(-1, 4),
                                    np.array(gt_labels, dtype=np.float32).reshape(-1, 1)], axis=1)
        print(all_boxes)

        # 每个类进行nms
        tmp_boxes = []
        tmp_labels = []
        for label in [1, 2]:
            idx = np.where(all_boxes[:, 4] == label)[0]
            if len(idx) > 0:
                boxes_thisclass = all_boxes[idx, :4]
                labels_thisclass = all_boxes[idx, 4:5]
                dets = np.concatenate([boxes_thisclass.astype(np.float32),
                                       0.99 * np.ones_like(idx, dtype=np.float32).reshape([-1, 1])], axis=1)
                keep = py_cpu_nms(dets, thresh=0.5)
                tmp_boxes.append(boxes_thisclass[keep])
                tmp_labels.append(labels_thisclass[keep])
        gt_boxes = np.concatenate(tmp_boxes)
        gt_labels = np.concatenate(tmp_labels).reshape([-1, 1])

        if len(gt_boxes) == 0 or len(gt_boxes) != len(gt_labels):
            continue

        count += len(all_boxes)

        # new_gt_boxes = []
        # new_gt_labels = []
        # for label in [1, 2]:
        #     inds = np.where(gt_labels==label)[0]
        #     if label == 1:
        #         gt_w = gt_boxes[inds, 2] - gt_boxes[inds, 0]
        #         gt_h = gt_boxes[inds, 3] - gt_boxes[inds, 1]
        #         inds2 = np.where((gt_w > 100) | (gt_h > 100))[0]
        #         new_gt_boxes.append(gt_boxes[inds[inds2]])
        #         new_gt_labels.append(gt_labels[inds[inds2]])
        #     else:
        #         new_gt_boxes.append(gt_boxes[inds])
        #         new_gt_labels.append(gt_labels[inds])
        # gt_boxes = np.concatenate(new_gt_boxes)
        # gt_labels = np.concatenate(new_gt_labels)

        # 写入新的xml文件
        preds = np.concatenate([gt_boxes.reshape([-1, 4]), gt_labels.reshape([-1, 1])], axis=1)
        save_xml_filename = os.path.join(new_gt_dir, file_prefix+'_gt_new.xml')
        save_predictions_to_envi_xml(preds, save_xml_filename,
                                     gdal_proj_info=projection_esri,
                                     gdal_trans_info=geotransform)

    print('count: ', count)
