
import cv2
from natsort import natsorted
import glob,os
import xml.dom.minidom


def main(subset='train'):
    if subset == 'train':
        subsize = 5120
        gap = 512
        orig_img_path = '/media/ubuntu/Temp/gd/data/aerial/%d_%d/' % (subsize, gap)
    elif subset == 'val':
        subsize = 5120
        gap = 128
        orig_img_path = '/media/ubuntu/Temp/gd/data/aerial2/%d_%d/' % (subsize, gap)
    anno_path = '/home/ubuntu/Downloads/Annotations/%s/' % subset
    save_root = orig_img_path.replace('%d_%d'%(subsize,gap), '%d_%d_shown'%(subsize,gap))

    for p in [save_root]:
        if not os.path.exists(p):
            os.makedirs(p)
    xmlfiles = natsorted(os.listdir(anno_path))

    for xmlfile in xmlfiles:
        DomTree = xml.dom.minidom.parse(anno_path + xmlfile)
        annotation = DomTree.documentElement

        filenamelist = annotation.getElementsByTagName('filename')  # [<DOM Element: filename at 0x381f788>]
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        print(filename, '=' * 50)

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

        print(boxes)
        print(labels)

        folder_name = filename.split('_')[0]
        orig_img_filename = os.path.join(orig_img_path, folder_name, filename)
        im = cv2.imread(orig_img_filename)

        if im is not None and len(boxes) > 0:
            for box, label in zip(boxes, labels):  # xyxy, score, label
                xmin, ymin, xmax, ymax = box
                label = int(label) - 1
                cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              color=(0, 255, 0),
                              thickness=2, lineType=cv2.LINE_AA)
                # label_txt = f'{names[label]} {conf:.2f}'
                # cv2.putText(im0, label_txt, (int(xmin), int(ymin) - 2), 0, tl / 3,
                #             color=[225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        if im is not None:
            cv2.imwrite('%s/%s'%(save_root, filename.replace('.tif','.jpg')), im)


if __name__ == '__main__':
    main(subset='train')
    main(subset='val')




















