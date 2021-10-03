import sys, os, glob, shutil
import numpy as np
import cv2
import sklearn
from sklearn.cluster import KMeans

save_dir = 'E:/Downloads/BData/generated/'
for label_name in ['normal', 'defective', 'shown']:
    if not os.path.exists(os.path.join(save_dir, label_name)):
        os.makedirs(os.path.join(save_dir, label_name))

im0 = cv2.imread('E:/Downloads/BData/images_with_alpha/038_aug_0.png', cv2.IMREAD_UNCHANGED)
im1 = cv2.imread('E:/Downloads/BData/images_with_alpha/038_aug_1.png', cv2.IMREAD_UNCHANGED)
print(im0.shape, im1.shape)
print(np.unique(im0[:,:,3]), np.unique(im1[:,:,3]))

colors_images = [cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                 for filename in glob.glob(os.path.join('E:/Downloads/BData/insulator_colors/*.png'))]
color_centers = []
n_clusters = 20
for im in colors_images:
    print(im.shape)
    cluster = KMeans(n_clusters=n_clusters)
    cluster.fit(im.reshape(-1, 3))
    print('cluster centers: ', cluster.cluster_centers_)
    new_cluster = []
    for center in cluster.cluster_centers_:
        if np.any(center < 240):
            new_cluster.append(center)
    color_centers.append(np.array(new_cluster))
    print(cluster.labels_.shape)
print(color_centers)
# sys.exit(-1)

for index in range(100):
    boxes = []
    ims = [im1]
    h, w = im1.shape[:2]
    label = [1]
    xmins = [0]
    for step in range(20):
        if np.random.rand() < 0.2:
            ims.append(im0)
            boxes.append([w, 0, w + im0.shape[1], im0.shape[0]])
            xmins.append(w)
            w += im0.shape[1]
            label.append(0)
        else:
            ims.append(im1)
            xmins.append(w)
            w += im1.shape[1]
            label.append(1)
    ims.append(im1)
    xmins.append(w)
    w += im1.shape[1]
    label.append(1)

    im = np.concatenate(ims, axis=1)
    im[:, :, :3] = cv2.medianBlur(im[:, :, :3], np.random.choice([3, 5, 7, 9]))

    if np.all(label):
        print('normal insuator')
        cv2.imwrite(os.path.join(save_dir, 'normal', '%03d.jpg' % index), im)
    else:
        print('defective insulator')
        cv2.imwrite(os.path.join(save_dir, 'defective', '%03d.jpg' % index), im)

    bgr, alpha = im[:, :, :3].copy(), im[:, :, 3:]
    print(bgr.shape, alpha.shape)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        print(xmin, ymin, xmax, ymax)
        cv2.rectangle(bgr, (xmin + 1, ymin + 1), (xmax - 1, ymax - 1), color=(0, 0, 255), thickness=2)
        cv2.imwrite(os.path.join(save_dir, 'shown', '%03d.png' % index),
                    np.concatenate([bgr, alpha], axis=2))

    # find all the 1->0, then find the right nearest 0->1
    print('label', label)
    inds = []
    ind = 0
    while ind < len(label) - 1:
        if label[ind] == 1 and label[ind + 1] == 0:
            s1 = ind
            ind2 = ind + 1
            while label[ind2] != 1:
                ind2 += 1
            s2 = ind2

            inds.append([s1, s2])

            ind = ind2
        else:
            ind += 1
    print(inds)
    for s1, s2 in inds:
        print(xmins[s1], 0, xmins[s2], 0)

    bgr, alpha = im[:, :, :3].copy(), im[:, :, 3:]
    print(bgr.shape, alpha.shape)
    for s1, s2 in inds:
        cv2.rectangle(bgr,
                      (xmins[s1] + np.random.randint(5, 15), np.random.randint(5, 10)),
                      (xmins[s2] + im0.shape[1] - np.random.randint(5, 15), im0.shape[0] - np.random.randint(5, 10)),
                      color=(0, 0, 255), thickness=2)
        cv2.imwrite(os.path.join(save_dir, 'shown', '%03d_merged.png' % index),
                    bgr)

    if np.random.rand() > 0.5:
        print('colored')
        predefined_cluster = color_centers[np.random.choice(np.arange(len(color_centers)))].astype(np.uint8)
        im_new = im.copy()
        bgr, alpha = im_new[:, :, :3], im_new[:, :, 3]
        print('alpha unique',np.unique(alpha))
        cluster = KMeans(n_clusters=predefined_cluster.shape[0])
        X = bgr[np.where(alpha)].reshape(-1, 3)
        print(X[:10, :])
        cluster.fit(X)
        labels = cluster.labels_
        print('cluster center', cluster.cluster_centers_)
        new_bgr = np.zeros((len(labels), 3), dtype=np.uint8)
        for c in range(predefined_cluster.shape[0]):
            new_bgr[labels == c, :] = predefined_cluster[np.random.choice(np.arange(predefined_cluster.shape[0]))]
        bgr[np.where(alpha)] = new_bgr + np.random.randint(-30, 30, size=(len(labels), 3))
        bgr[bgr < 0] = 0
        bgr[bgr > 255] = 255
        bgr = cv2.GaussianBlur(bgr, ksize=(np.random.choice([3,5,7]), np.random.choice([3,5,7])),
                               sigmaX=np.random.choice([0.1, 1, 10]))
        print('bgr', bgr.shape)
        im_new = np.concatenate([bgr, alpha[:, :, None]], axis=2)
        cv2.imwrite(os.path.join(save_dir, 'shown', '%03d_colored.png' % index),
                    np.concatenate([bgr, alpha[:, :, None]], axis=2))

        bgr = bgr.copy()
        for s1, s2 in inds:
            cv2.rectangle(bgr,
                          (xmins[s1] + np.random.randint(5, 15), np.random.randint(5, 10)),
                          (
                              xmins[s2] + im0.shape[1] - np.random.randint(5, 15),
                              im0.shape[0] - np.random.randint(5, 10)),
                          color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(save_dir, 'shown', '%03d_colored_merged.png' % index),
                        bgr)
