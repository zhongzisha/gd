import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from osgeo import gdal, osr
import psutil
import io
import json
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publisher
import pickle
import zlib

from mmdet.apis import inference_detector, init_detector
from myutils import py_cpu_nms, compute_offsets, LoadImages
from myutils import save_predictions_to_envi_xml_and_shp as save_predictions_to_envi_xml

from common import load_msg, send_msg


class Worker:
    def __init__(self):
        weights = './assets/det_model.pth'
        config = './assets/det_config.py'
        self.save_root = './results/'

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.device = "cuda:0"
            self.batchsize = 32
        else:
            self.device = "cpu"
            self.batchsize = 64

        self.model = init_detector(config, weights, device=self.device)
        self.names = {0: 'GanTa', 1: 'JueYuanZi'}
        self.colors = {0: [0, 0, 255], 1: [100, 255, 255]}

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

    def run(self, client, data_dict):
        device = torch.device(self.device)
        imgsz = 1024
        gap = 256
        gt_gap = 128
        big_subsize = 10240
        batchsize = self.batchsize
        score_thr = 0.1
        hw_thr = 10

        tiffile = data_dict['tiffile']
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')
        print(file_prefix)
        save_dir = os.path.join(self.save_root, file_prefix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        projection = ds.GetProjection()
        projection_sr = osr.SpatialReference(wkt=projection)
        projection_esri = projection_sr.ExportToWkt(["FORMAT=WKT1_ESRI"])
        geotransform = ds.GetGeoTransform()
        xOrigin = geotransform[0]
        yOrigin = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        orig_height, orig_width = ds.RasterYSize, ds.RasterXSize

        # 先计算可用内存，如果可以放得下，就不用分块了
        avaialble_mem_bytes = psutil.virtual_memory().available
        if False:  # orig_width * orig_height * ds.RasterCount < 0.8 * avaialble_mem_bytes:
            offsets = [[0, 0, orig_width, orig_height]]
        else:
            # 根据big_subsize计算子块的起始偏移
            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize,
                                      gap=2 * gt_gap)
        print('offsets: ', offsets)
        num_offsets = len(offsets)
        percent_steps = np.array_split(np.arange(90), num_offsets)
        print('percent_steps', percent_steps)

        all_preds = []
        for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up
            # sub_width = min(orig_width, big_subsize)
            # sub_height = min(orig_height, big_subsize)
            # if xoffset + sub_width > orig_width:
            #     sub_width = orig_width - xoffset
            # if yoffset + sub_height > orig_height:
            #     sub_height = orig_height - yoffset

            print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
            dataset = LoadImages(gdal_ds=ds, xoffset=xoffset, yoffset=yoffset,
                                 width=sub_width, height=sub_height,
                                 batchsize=batchsize, subsize=imgsz, gap=gap,
                                 return_list=True)
            if len(dataset) == 0:
                continue

            print('forward inference')
            sub_preds = []
            for img in dataset:

                result = inference_detector(self.model, img)

                if isinstance(result, tuple):
                    bbox_results, segm_results = result
                else:
                    bbox_results, segm_results = result, None

                pred_per_image = []
                for bbox_result in bbox_results:

                    bboxes = np.concatenate(bbox_result, axis=0)

                    pred_labels = [
                        np.full(bbox.shape[0], i, dtype=np.int32)
                        for i, bbox in enumerate(bbox_result)
                    ]
                    pred_labels = np.concatenate(pred_labels, axis=0)

                    if score_thr > 0:
                        assert bboxes.shape[1] == 5
                        scores = bboxes[:, -1]
                        inds = scores > score_thr
                        bboxes = bboxes[inds, :]
                        pred_labels = pred_labels[inds]

                    # 过滤那些框的宽高不合理的框
                    if hw_thr > 0 and len(bboxes) > 0:
                        ws = bboxes[:, 2] - bboxes[:, 0]
                        hs = bboxes[:, 3] - bboxes[:, 1]
                        inds = np.where((hs > hw_thr) & (ws > hw_thr))[0]
                        bboxes = bboxes[inds, :]
                        pred_labels = pred_labels[inds]

                    pred = torch.from_numpy(
                        np.concatenate([bboxes, pred_labels.reshape(-1, 1)], axis=1))  # xyxy,score,cls

                    # pred is [xyxy, conf, pred_label]
                    pred_per_image.append(pred)

                sub_preds += pred_per_image

            # 合并子图上的检测，再次进行nms
            newpred = []
            for det, (x, y) in zip(sub_preds, dataset.start_positions):
                if len(det):
                    det[:, [0, 2]] += x
                    det[:, [1, 3]] += y
                    newpred.append(det)

            if len(newpred) > 0:
                sub_preds = torch.cat(newpred)
            else:
                sub_preds = []

            if len(sub_preds) > 0:
                sub_preds[:, [0, 2]] += xoffset
                sub_preds[:, [1, 3]] += yoffset
                all_preds.append(sub_preds)

            del dataset.img0
            del dataset
            import gc
            gc.collect()

            data_dict["percent"] = int(percent_steps[oi][-1])
            send_msg("gd/detection_results", msg_dict=data_dict)

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)  # xmin, ymin, xmax, ymax, score, label

        if len(all_preds) == 0:
            pass

        # 过滤那些框的宽高不合理的框
        if hw_thr > 0 and len(all_preds) > 0:
            ws = all_preds[:, 2] - all_preds[:, 0]
            hs = all_preds[:, 3] - all_preds[:, 1]
            inds = np.where((hs > hw_thr) & (ws > hw_thr))[0]
            all_preds = all_preds[inds, :]

        all_preds_before = all_preds.clone()

        all_preds = all_preds.to(device)
        # 只保留0.5得分的框
        # all_preds = all_preds[all_preds[:, 4] >= 0.5, :]
        # TODO zzs 这里需要在全图范围内再来一次nms
        # 每个类进行nms
        tmp_preds = []
        all_preds_cpu = all_preds.cpu().numpy()
        for label in [0, 1, 2, 3]:
            idx = np.where(all_preds_cpu[:, 5] == label)[0]
            if len(idx) > 0:
                # if label == 0:
                #     pw = all_preds_cpu[idx, 2] - all_preds_cpu[idx, 0]
                #     ph = all_preds_cpu[idx, 3] - all_preds_cpu[idx, 1]
                #     inds = np.where((pw >= 100) | (ph >= 100))[0]
                #     valid_inds = idx[inds]
                # elif label == 1:
                #     pw = all_preds_cpu[idx, 2] - all_preds_cpu[idx, 0]
                #     ph = all_preds_cpu[idx, 3] - all_preds_cpu[idx, 1]
                #     inds = np.where((pw < 100) & (ph < 100))[0]
                #     valid_inds = idx[inds]
                valid_inds = idx
                dets = all_preds_cpu[valid_inds, :5]
                keep = py_cpu_nms(dets, thresh=0.5)
                tmp_preds.append(all_preds[valid_inds[keep]])
        all_preds = torch.cat(tmp_preds)

        data_dict["percent"] = int(95)
        send_msg("gd/detection_results", msg_dict=data_dict)

        if len(all_preds) > 0:
            # 对杆塔目标进行聚类合并，减少一个杆塔多个预测框
            all_preds_cpu = all_preds.cpu().numpy()

            idx = np.where(all_preds_cpu[:, 5] == 0)[0]  # 0:小杆塔，1:中杆塔，2:大杆塔
            all_preds_small = all_preds_cpu[idx, :]
            all_preds_small = all_preds_small[all_preds_small[:, 4] > 0.5]  # TODO 这里阈值需要设定
            if len(all_preds_small) == 0:
                all_preds_small = np.empty((0, 6), dtype=all_preds_cpu.dtype)

            idx = np.where(all_preds_cpu[:, 5] == 1)[0]  # 0:小杆塔，1:中杆塔，2:大杆塔
            all_preds_mid = all_preds_cpu[idx, :]
            all_preds_mid = all_preds_mid[all_preds_mid[:, 4] > 0.5]  # TODO 这里阈值需要设定
            if len(all_preds_mid) == 0:
                all_preds_mid = np.empty((0, 6), dtype=all_preds_cpu.dtype)

            idx = np.where(all_preds_cpu[:, 5] == 2)[0]  # 0:小杆塔，1:中杆塔，2:大杆塔
            all_preds_0 = all_preds_cpu[idx, :]
            idx = np.where(all_preds_cpu[:, 5] == 3)[0]  # 3:绝缘子
            all_preds_1 = all_preds_cpu[idx, :]
            all_preds_1 = all_preds_1[all_preds_1[:, 4] > 0.5]  # TODO 这里阈值需要设定
            if len(all_preds_0) > 0:
                print('before cluster: %d' % (len(all_preds_0)))
                # 只对杆塔进行聚类
                xmin, ymin, xmax, ymax = np.split(all_preds_0[:, :4], 4, axis=1)
                xc = (xmin + xmax) / 2
                yc = (ymin + ymax) / 2
                ws = xmax - xmin
                hs = ymax - ymin
                estimator = DBSCAN(eps=max(ws.mean(), hs.mean()) * 1.5, min_samples=1)
                X = np.concatenate([xc, yc], axis=1)  # N x 2

                estimator.fit(X)
                ##初始化一个全是False的bool类型的数组
                core_samples_mask = np.zeros_like(estimator.labels_, dtype=bool)
                '''
                   这里是关键点(针对这行代码：xy = X[class_member_mask & ~core_samples_mask])：
                   db.core_sample_indices_  表示的是某个点在寻找核心点集合的过程中暂时被标为噪声点的点(即周围点
                   小于min_samples)，并不是最终的噪声点。在对核心点进行联通的过程中，这部分点会被进行重新归类(即标签
                   并不会是表示噪声点的-1)，也可也这样理解，这些点不适合做核心点，但是会被包含在某个核心点的范围之内
                '''
                core_samples_mask[estimator.core_sample_indices_] = True

                ##每个数据的分类
                cluster_lables = estimator.labels_
                unique_labels = set(cluster_lables)
                ##分类个数：lables中包含-1，表示噪声点
                n_clusters_ = len(np.unique(cluster_lables)) - (1 if -1 in cluster_lables else 0)

                newpred = []
                for k in unique_labels:
                    # ##-1表示噪声点,这里的k表示黑色
                    # if k == -1:
                    #     col = 'k'

                    ##生成一个True、False数组，lables == k 的设置成True
                    class_member_mask = (cluster_lables == k)
                    indices = class_member_mask & core_samples_mask

                    ##两个数组做&运算，找出即是核心点又等于分类k的值  markeredgecolor='k',
                    # xy = X[class_member_mask & core_samples_mask]
                    # plt.plot(xy[:, 0], xy[:, 1], 'o', c=col, markersize=14)
                    '''
                       1)~优先级最高，按位对core_samples_mask 求反，求出的是噪音点的位置
                       2)& 于运算之后，求出虽然刚开始是噪音点的位置，但是重新归类却属于k的点
                       3)对核心分类之后进行的扩展
                    '''
                    # xy = X[class_member_mask & ~core_samples_mask]
                    # plt.plot(xy[:, 0], xy[:, 1], 'o', c=col, markersize=6)

                    tmppred = all_preds_0[indices]
                    xmin = np.min(tmppred[:, 0])
                    ymin = np.min(tmppred[:, 1])
                    xmax = np.max(tmppred[:, 2])
                    ymax = np.max(tmppred[:, 3])
                    score = np.max(tmppred[:, 4])
                    print(k, score)
                    newpred.append([xmin, ymin, xmax, ymax, score, 2])
                if len(newpred) > 0:
                    all_preds_0_clustered = np.array(newpred)
                    # 使用规则去掉不合理的杆塔 TODO 可能需要修改规则
                    ws = all_preds_0_clustered[:, 2] - all_preds_0_clustered[:, 0]
                    hs = all_preds_0_clustered[:, 3] - all_preds_0_clustered[:, 1]
                    wsmean, hsmean = ws.mean(), hs.mean()
                    thres = min(wsmean, hsmean)
                    # TODO 这里的规则就是检测出来的杆塔长宽要满足一个条件，否则去掉那个检测
                    inds = np.where((ws > 0.25 * thres) & (hs > 0.25 * thres))[0]
                    all_preds_0_clustered = all_preds_0_clustered[inds]
                    all_preds_0_clustered = all_preds_0_clustered[all_preds_0_clustered[:, 4] > 0.15]  # TODO 这里阈值需要设定

                else:
                    all_preds_0_clustered = []

                print('after cluster: %d' % (len(all_preds_0_clustered)))
            else:
                all_preds_0_clustered = []

            if len(all_preds_0_clustered) > 0 and len(all_preds_1) > 0:
                all_preds_cpu = np.concatenate([all_preds_0_clustered, all_preds_1], axis=0)
            elif len(all_preds_0_clustered) > 0 and len(all_preds_1) == 0:
                all_preds_cpu = all_preds_0_clustered
            elif len(all_preds_0_clustered) == 0 and len(all_preds_1) > 0:
                all_preds_cpu = all_preds_1
            else:
                all_preds_cpu = np.empty((0, 6), dtype=all_preds_cpu.dtype)

            all_preds_cpu = np.concatenate([all_preds_cpu, all_preds_small, all_preds_mid], axis=0)
            all_preds = torch.from_numpy(all_preds_cpu).to(all_preds.device)

        # remove 0,1 in all_preds
        # remove 1,2 in gt_labels
        if len(all_preds) > 0:
            inds = torch.where(all_preds[:, 5] > 1)
            all_preds = all_preds[inds]
            all_preds[all_preds[:, 5] == 2, 5] = 0
            all_preds[all_preds[:, 5] == 3, 5] = 1

        save_filename = os.path.join(save_dir, file_prefix + '.xml')
        save_predictions_to_envi_xml(preds=all_preds.cpu(),
                                     save_xml_filename=os.path.join(save_dir, file_prefix + '.xml'),
                                     gdal_proj_info=projection_esri,
                                     gdal_trans_info=geotransform,
                                     names=self.names,
                                     colors=self.colors,
                                     spatialreference=projection_sr)
        save_predictions_to_envi_xml(preds=all_preds_before.cpu(),
                                     save_xml_filename=os.path.join(save_dir, file_prefix + '_before.xml'),
                                     gdal_proj_info=projection_esri,
                                     gdal_trans_info=geotransform,
                                     names=self.names,
                                     colors=self.colors,
                                     spatialreference=projection_sr)

        data_dict["percent"] = int(100)
        data_dict["save_filename"] = save_filename
        send_msg("gd/detection_results", msg_dict=data_dict)


worker = Worker()


def on_connect(client, userdata, flags, rc):
    print('connected with result code ' + str(rc))
    client.subscribe("gd/detection")


def on_message(client, userdata, msg):
    data_dict = load_msg(msg)
    print('topic', msg.topic)
    print(data_dict)

    worker.run(client, data_dict)


mqtt_client = mqtt.Client("mqtt client")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect("10.0.7.65", 21883, 60)

mqtt_client.loop_forever()
