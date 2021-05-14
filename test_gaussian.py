import sys,os,glob
import numpy as np
import cv2
from matplotlib import pyplot as plt


def gauss_map(size_x, size_y=None, sigma_x=5, sigma_y=None):
    if size_y == None:
        size_y = size_x
    if sigma_y == None:
        sigma_y = sigma_x

    assert isinstance(size_x, int)
    assert isinstance(size_y, int)

    x0 = size_x // 2
    y0 = size_y // 2

    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:, np.newaxis]

    x -= x0
    y -= y0

    exp_part = x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)
    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)



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

# H, W = 10, 20
#
# gauss = gauss_map(size_x=W, size_y=H, sigma_x=W, sigma_y=H)
# gauss_01 = (gauss - np.min(gauss)) / (np.max(gauss) - np.min(gauss))
# gauss_01 = (255 * gauss_01).astype(np.uint8)
# plt.imshow(gauss_01)
# plt.show()

# alpha = alpha_map(256, 256, 30, 40, 100, 100)
# plt.imshow(alpha)
# plt.show()

# im = np.zeros((100, 100, 3), dtype=np.uint8)
# im = cv2.line(im, (-50, 50), (50, 0), color=(255, 255, 255), thickness=5)
# plt.imshow(im)
# plt.show()


import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


ia.seed(1)

image = ia.quokka(size=(256, 256))
kps = KeypointsOnImage([
    Keypoint(x=65, y=100),
    Keypoint(x=75, y=200),
    Keypoint(x=100, y=100),
    Keypoint(x=200, y=80)
], shape=image.shape)


for degree in range(10, 360, 10):
    seq = iaa.Sequential([
        iaa.Affine(
            rotate=degree,
            fit_output=True
        ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    ])

    # Augment keypoints and images.
    image_aug, kps_aug = seq(image=image, keypoints=kps)

    # print coordinates before/after augmentation (see below)
    # use after.x_int and after.y_int to get rounded integer coordinates
    for i in range(len(kps.keypoints)):
        before = kps.keypoints[i]
        after = kps_aug.keypoints[i]
        print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
            i, before.x, before.y, after.x, after.y)
        )

    # image with keypoints before/after augmentation (shown below)
    image_before = kps.draw_on_image(image, size=7)
    image_after = kps_aug.draw_on_image(image_aug, size=7)

    cv2.imwrite('H:/tmp/degree_%d.png'%degree, image_after)
