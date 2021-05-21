import sys,os,glob
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import math
from PIL import Image
import shutil

if __name__ == '__main__':

    files = glob.glob('E:/VesselSeg-Pytorch/gd/gd_line_seg_U_Net_bs=4_lr=0.0001/big_results_noBoxes/test_*_LineSeg_result.png')
    for ii, filename in enumerate(files):
        new_filename = os.path.join('E:/VesselSeg-Pytorch/gd/gd_line_seg_U_Net_bs=4_lr=0.0001/big_results_noBoxes/', '%d.png'%ii)
        if not os.path.exists(new_filename):
            shutil.copyfile(filename, new_filename)
        filename1 = filename.replace('U_Net', 'Dense_Unet')
        new_filename1 = os.path.join('E:/VesselSeg-Pytorch/gd/gd_line_seg_Dense_Unet_bs=4_lr=0.0001/big_results_noBoxes/', '%d.png'%ii)
        if not os.path.exists(new_filename1):
            shutil.copyfile(filename1, new_filename1)
        filename = os.path.join('E:/VesselSeg-Pytorch/gd/gd_line_seg_U_Net_bs=4_lr=0.0001/big_results_noBoxes/', '%d.png'%ii)
        prefix = os.path.basename(filename).replace('.png', '')
        # dst = cv2.imread(filename, 0)
        # print(dst.shape)
        # lines = cv2.HoughLines(dst, 1, np.pi/180, 150, None, 0, 0)
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        #         cv2.line(dst, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
        #
        # cv2.imencode('.png', dst)[1].tofile(filename.replace('.png', '_h1.png'))
        # del dst

        pred_unet = cv2.imread(filename, 0)
        pred_dense_unet = cv2.imread(filename.replace('U_Net', 'Dense_Unet'), 0)
        # pred_unet = np.array(Image.open(filename))
        # pred_dense_unet = np.array(Image.open(filename.replace('gd_line_UNet', 'gd_line_DenseUNet')))
        # pred_unet = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), 0)
        # pred_dense_unet = cv2.imdecode(np.fromfile(filename.replace('gd_line_UNet', 'gd_line_DenseUNet'), dtype=np.uint8), 0)
        pred = pred_unet.astype(np.short)
        pred_dense_unet = pred_dense_unet.astype(np.short)
        pred += pred_dense_unet
        del pred_dense_unet
        pred = pred > 128
        cv2.imencode('.png', pred.astype(np.uint8) * 255)[1].tofile(filename.replace('.png', '_combine.png'))
        print(pred.shape)

        se = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(10, 10))
        pred_dilated = cv2.dilate(pred.astype(np.uint8), se)
        cv2.imencode('.png', pred_dilated.astype(np.uint8) * 255)[1].tofile(filename.replace('.png', '_dilated.png'))

        pred = pred.astype(np.uint8)
        linesP = cv2.HoughLinesP(pred, 1, np.pi / 180, 150, None, 100, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(pred, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imencode('.png', pred)[1].tofile(filename.replace('.png', '_h2.png'))
        del pred

        print(filename, ' done')
