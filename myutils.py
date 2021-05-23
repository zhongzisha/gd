import os
import copy
import glob
import numpy as np
import torch
import cv2
import xml.dom.minidom
from pathlib import Path
import matplotlib.pyplot as plt
from numba import jit
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


@jit
def box_iou_np(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


@jit
def box_intersection_np(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    # # compute the area of both the prediction and ground-truth rectangles
    # boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    # boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    #
    # iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return interArea

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def ap_per_class1(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


class ConfusionMatrix1:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))

# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


class LoadImages:  # for inference
    def __init__(self, gdal_ds, xoffset, yoffset, width, height,
                 batchsize=32, subsize=1024, gap=128, stride=32,
                 return_list=False, is_nchw=True, return_positions=False):
        self.batchsize = batchsize
        self.subsize = subsize
        self.gap = gap
        self.slide = subsize - gap
        self.return_list = return_list
        self.is_nchw = is_nchw
        self.return_positions = return_positions

        self.height = height
        self.width = width
        self.img0 = np.zeros((height, width, 3), dtype=np.uint8)  # RGB format
        print('11111111111', xoffset, yoffset, width, height)
        for b in range(3):
            band = gdal_ds.GetRasterBand(b + 1)
            self.img0[:, :, b] = band.ReadAsArray(xoffset, yoffset, win_xsize=width, win_ysize=height)
            # self.img0[:, :, b] = band.ReadRaster(xoffset, yoffset, width, height)

        self.start_positions = []
        left, up = 0, 0
        while left < width:
            # if left + self.subsize >= width:
            #     left = max(width - self.subsize, 0)
            up = 0
            while up < height:
                # if up + self.subsize >= height:
                #     up = max(height - self.subsize, 0)
                right = min(left + self.subsize, width - 1)
                down = min(up + self.subsize, height - 1)

                subimg = self.img0[up: (up + self.subsize), left: (left + self.subsize)]
                minval = subimg.min()
                maxval = subimg.max()
                if maxval > minval:
                    self.start_positions.append([left, up])

                if up + self.subsize >= height:
                    break
                else:
                    up = up + self.slide
            if left + self.subsize >= width:
                break
            else:
                left = left + self.slide

        print('start_positions: ', self.start_positions[:10])
        print('len(start_positions): ', len(self.start_positions))
        print('height', self.height)
        print('width', self.width)
        print('shape', self.img0.shape)
        # subimg = copy.deepcopy(self.img0[:5120, :5120, :])
        # cv2.imwrite('1.jpg', subimg)
        # subimg = copy.deepcopy(self.img0[5120 * 2:(5120 * 3), 5120 * 2:(5120 * 3), :])
        # cv2.imwrite('2.jpg', subimg)
        # subimg = copy.deepcopy(self.img0[5120 * 3:(5120 * 4), 5120 * 2:(5120 * 3), :])
        # cv2.imwrite('3.jpg', subimg)
        # subimg = copy.deepcopy(self.img0[:5120, :5120, :])
        # cv2.imwrite('11.jpg', subimg[:, :, ::-1])
        # subimg = copy.deepcopy(self.img0[5120 * 2:(5120 * 3), 5120 * 2:(5120 * 3), :])
        # cv2.imwrite('21.jpg', subimg[:, :, ::-1])
        # subimg = copy.deepcopy(self.img0[5120 * 3:(5120 * 4), 5120 * 2:(5120 * 3), :])
        # cv2.imwrite('31.jpg', subimg[:, :, ::-1])
        self.stride = stride
        self.nf = len(self.start_positions)  # number of sub images

        self.mode = 'image'
        self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= self.nf:
            raise StopIteration

        batch_imgs = []
        batch_positions = []
        if not self.return_list:
            for bi in range(self.batchsize):
                if self.count == self.nf:
                    break
                left, up = self.start_positions[self.count]
                subimg = copy.deepcopy(self.img0[up: (up + self.subsize), left: (left + self.subsize)])
                h, w, c = np.shape(subimg)
                outimg = np.zeros((self.subsize, self.subsize, 3), dtype=subimg.dtype)
                outimg[0:h, 0:w, :] = subimg
                # batch_imgs.append(outimg.transpose((2, 0, 1)))  # RGB, to 3x416x416
                batch_imgs.append(outimg)
                batch_positions.append([left, up])
                self.count += 1

            if self.is_nchw:
                batch_imgs = np.stack(batch_imgs).transpose((0, 3, 1, 2))  # B x H x W x 3 --> B3HW
            else:
                batch_imgs = np.stack(batch_imgs)
            batch_imgs = np.ascontiguousarray(batch_imgs)  #
        else:
            for bi in range(self.batchsize):
                if self.count == self.nf:
                    break
                left, up = self.start_positions[self.count]
                subimg = copy.deepcopy(self.img0[up: (up + self.subsize), left: (left + self.subsize)])
                h, w, c = np.shape(subimg)
                outimg = np.zeros((self.subsize, self.subsize, 3), dtype=subimg.dtype)
                outimg[0:h, 0:w, :] = subimg[:, :, ::-1]
                # batch_imgs.append(outimg.transpose((2, 0, 1)))  # RGB, to 3x416x416
                batch_imgs.append(outimg)
                batch_positions.append([left, up])
                self.count += 1

        if self.return_positions:
            return batch_imgs, batch_positions
        else:
            return batch_imgs

    def __len__(self):
        return self.nf  # number of files


def get_gt_boxes(gt_xml_filename):
    DomTree = xml.dom.minidom.parse(gt_xml_filename)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')  # [<DOM Element: filename at 0x381f788>]
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    # print(filename, '=' * 50)

    boxes = []
    labels = []
    for objects in objectlist:
        # print objects

        namelist = objects.getElementsByTagName('name')
        if len(namelist) == 0:
            continue

        # print 'namelist:',namelist
        objectname = namelist[0].childNodes[0].data
        labels.append(int(objectname))

        bndbox = objects.getElementsByTagName('bndbox')[0]
        x1_list = bndbox.getElementsByTagName('xmin')
        x1 = int(float(x1_list[0].childNodes[0].data))
        y1_list = bndbox.getElementsByTagName('ymin')
        y1 = int(float(y1_list[0].childNodes[0].data))
        x2_list = bndbox.getElementsByTagName('xmax')
        x2 = int(float(x2_list[0].childNodes[0].data))
        y2_list = bndbox.getElementsByTagName('ymax')
        y2 = int(float(y2_list[0].childNodes[0].data))
        w = x2 - x1
        h = y2 - y1

        boxes.append([x1, y1, x2, y2])
    return boxes, labels


def save_predictions_to_envi_xml(preds, save_xml_filename, gdal_proj_info, gdal_trans_info,
                                 names=None, colors=None, is_line=False):
    if names is None:
        names = {0: '1', 1: '2'}
    if colors is None:
        colors = {0: "255,0,0", 1: "0,0,255"}

    def get_coords_str(xmin, ymin, xmax, ymax):
        # [xmin, ymin]
        x1 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
        y1 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]

        # [xmax, ymax]
        x3 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
        y3 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]

        if is_line:
            return " ".join(['%.6f' % val for val in [x1, y1, x3, y3]])
        else:
            # [xmax, ymin]
            x2 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
            y2 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]
            # [xmin, ymax]
            x4 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
            y4 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]

            return " ".join(['%.6f' % val for val in [x1, y1, x2, y2, x3, y3, x4, y4, x1, y1]])

    lines = ['<?xml version="1.0" encoding="UTF-8"?>\n<RegionsOfInterest version="1.0">\n']
    # names = {0: '1', 1: '2'}

    is_gt = False
    if len(preds) > 0:
        is_gt = len(preds[0]) == 5

    for current_label, label_name in names.items():
        lines1 = ['<Region name="%s" color="%s">\n' % (label_name, colors[current_label]),
                  '<GeometryDef>\n<CoordSysStr>%s</CoordSysStr>\n' % (
                      gdal_proj_info if gdal_proj_info != '' else 'none')]  # 这里不能有换行符

        count = 0
        for i, pred in enumerate(preds):
            if is_gt:
                xmin, ymin, xmax, ymax, label = pred
                label = int(label) - 1  # label==0: 杆塔, label==1: 绝缘子
            else:
                xmin, ymin, xmax, ymax, score, label = pred
                label = int(label)  # label==0: 杆塔, label==1: 绝缘子

            if label == current_label:
                coords_str = get_coords_str(xmin, ymin, xmax, ymax)

                if is_line:
                    lines1.append('<LineString>\n<Coordinates>\n')
                    lines1.append('%s\n' % (coords_str))
                    lines1.append('</Coordinates>\n</LineString>\n')
                else:
                    lines1.append('<Polygon>\n<Exterior>\n<LinearRing>\n<Coordinates>\n')
                    lines1.append('%s\n' % (coords_str))
                    lines1.append('</Coordinates>\n</LinearRing>\n</Exterior>\n</Polygon>\n')

                count += 1
        lines1.append('</GeometryDef>\n</Region>\n')

        if count > 0:
            lines.append(''.join(lines1))

    lines.append('</RegionsOfInterest>\n')

    with open(save_xml_filename, 'w') as fp:
        fp.writelines(lines)


def save_predictions_to_envi_xml_bak(preds, save_xml_filename, gdal_proj_info, gdal_trans_info,
                                     names=None, colors=None):
    if names is None:
        names = {0: '1', 1: '2'}
    if colors is None:
        colors = {0: "255,0,0", 1: "0,0,255"}

    def get_coords_str(xmin, ymin, xmax, ymax):
        x1 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
        y1 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]

        x2 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
        y2 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]

        x3 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
        y3 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]

        x4 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
        y4 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]
        return " ".join(['%.6f' % val for val in [x1, y1, x2, y2, x3, y3, x4, y4, x1, y1]])

    lines = ['<?xml version="1.0" encoding="UTF-8"?>\n<RegionsOfInterest version="1.0">\n']
    # names = {0: '1', 1: '2'}

    is_gt = False
    if len(preds) > 0:
        is_gt = len(preds[0]) == 5

    for current_label, label_name in names.items():
        lines1 = ['<Region name="%s" color="%s">\n' % (label_name, colors[current_label]),
                  '<GeometryDef>\n<CoordSysStr>%s</CoordSysStr>\n' % (
                      gdal_proj_info if gdal_proj_info != '' else 'none')]  # 这里不能有换行符

        count = 0
        for i, pred in enumerate(preds):
            if is_gt:
                xmin, ymin, xmax, ymax, label = pred
                label = int(label) - 1  # label==0: 杆塔, label==1: 绝缘子
            else:
                xmin, ymin, xmax, ymax, score, label = pred
                label = int(label)  # label==0: 杆塔, label==1: 绝缘子

            if label == current_label:
                coords_str = get_coords_str(xmin, ymin, xmax, ymax)

                lines1.append('<Polygon>\n<Exterior>\n<LinearRing>\n<Coordinates>\n')
                lines1.append('%s\n' % (coords_str))
                lines1.append('</Coordinates>\n</LinearRing>\n</Exterior>\n</Polygon>\n')
                count += 1
        lines1.append('</GeometryDef>\n</Region>\n')

        if count > 0:
            lines.append(''.join(lines1))

    lines.append('</RegionsOfInterest>\n')

    with open(save_xml_filename, 'w') as fp:
        fp.writelines(lines)


def compute_offsets(height, width, subsize, gap):
    slide = subsize - gap
    start_positions = []
    left, up = 0, 0
    while left < width:
        # if left + subsize >= width:
        #     left = max(width - subsize, 0)
        up = 0
        while up < height:
            # if up + subsize >= height:
            #     up = max(height - subsize, 0)
            right = min(left + subsize, width)
            down = min(up + subsize, height)
            sub_width = right - left
            sub_height = down - up

            start_positions.append([left, up, sub_width, sub_height])

            if up + subsize >= height:
                break
            else:
                up = up + slide
        if left + subsize >= width:
            break
        else:
            left = left + slide
    return start_positions


def load_gt(gt_xml_dir, gt_prefix, gt_subsize=5120, gt_gap=128):
    xmlfiles = glob.glob(gt_xml_dir + '/{}*.xml'.format(gt_prefix))
    all_boxes = []
    all_labels = []

    for xmlfile in xmlfiles:

        boxes, labels = get_gt_boxes(xmlfile)
        i, j = xmlfile.split(os.sep)[-1].replace('.xml', '').split('_')[1:3]
        up, left = (int(float(i)) - 1) * (gt_subsize - gt_gap), (int(float(j)) - 1) * (gt_subsize - gt_gap)
        if len(boxes) > 0:
            boxes = np.array(boxes)
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += up
            all_boxes.append(boxes)
            all_labels += labels

    all_boxes = np.concatenate(all_boxes)

    return all_boxes, all_labels


def load_gt_from_txt(filename):
    if not os.path.exists(filename):
        return [], []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    lines = np.array([line.strip().split(' ') for line in lines])
    boxes = lines[:, :4].astype(np.float32).tolist()
    labels = [int(float(val)) for val in lines[:, 4].flatten()]
    return boxes, labels


def load_gt_from_esri_xml(filename, gdal_trans_info, mapcoords2pixelcoords=True):
    if not os.path.exists(filename):
        return [], []
    DomTree = xml.dom.minidom.parse(filename)
    annotation = DomTree.documentElement
    regionlist = annotation.getElementsByTagName('Region')
    boxes = []
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
            xmin, ymin, xmax, ymax = coords[0], coords[1], coords[4], coords[5]  # this is the map coordinates
            # x1 = gdal_trans_info[0] + (xmin + 0.5) * gdal_trans_info[1] + (ymin + 0.5) * gdal_trans_info[2]
            # y1 = gdal_trans_info[3] + (xmin + 0.5) * gdal_trans_info[4] + (ymin + 0.5) * gdal_trans_info[5]
            # x3 = gdal_trans_info[0] + (xmax + 0.5) * gdal_trans_info[1] + (ymax + 0.5) * gdal_trans_info[2]
            # y3 = gdal_trans_info[3] + (xmax + 0.5) * gdal_trans_info[4] + (ymax + 0.5) * gdal_trans_info[5]
            if mapcoords2pixelcoords:
                x1 = int((xmin - xOrigin) / pixelWidth + 0.5)
                y1 = int((ymin - yOrigin) / pixelHeight + 0.5)
                x3 = int((xmax - xOrigin) / pixelWidth + 0.5)
                y3 = int((ymax - yOrigin) / pixelHeight + 0.5)
            else:
                x1, y1, x3, y3 = xmin, ymin, xmax, ymax

            boxes.append(np.array([x1, y1, x3, y3]))
            labels.append(int(float(label)))  # 0 is gan, 1 is jueyuanzi

    return boxes, labels


# 从envi的xml中获取polygons,这些多边形就是点组成
def load_gt_polys_from_esri_xml(filename, gdal_trans_info, mapcoords2pixelcoords=True):
    if not os.path.exists(filename):
        return [], []
    DomTree = xml.dom.minidom.parse(filename)
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
            if mapcoords2pixelcoords:
                points[:, 0] -= xOrigin
                points[:, 1] -= yOrigin
                points[:, 0] /= pixelWidth
                points[:, 1] /= pixelHeight
                points += 0.5
                points = points.astype(np.int32)

            polys.append(points)
            labels.append(int(float(label)))  # 0 is gan, 1 is jueyuanzi

    return polys, labels


def py_cpu_nms(dets, thresh):
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]

    # 这边的keep用于存放，NMS后剩余的方框
    keep = []

    # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    index = scores.argsort()[::-1]
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。

    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # every time the first is the biggst, and add it directly

        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)

        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h

        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx + 1]  # because index start from 1

    return keep


def alpha_map(size_x, size_y, sigma_x=None, sigma_y=None, xc=None, yc=None):
    if size_y is None:
        size_y = size_x
    if sigma_y is None:
        sigma_y = sigma_x

    assert isinstance(size_x, int)
    assert isinstance(size_y, int)

    if xc is None:
        xc = size_x // 2
    if yc is None:
        yc = size_y // 2

    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:, np.newaxis]

    x -= xc
    y -= yc

    exp_part = x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)
    alpha = 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)
    alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min())
    return np.stack([alpha, alpha, alpha], axis=-1)


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


# Function to distort image
def elastic_transform_v2(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)