# import the necessary packages
from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np


if False:
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=True,
                    help="path to the input source image")
    ap.add_argument("-r", "--reference", required=True,
                    help="path to the input reference image")
    ap.add_argument("-t", "--save_filename", required=True,
                    help="path to save")
    args = vars(ap.parse_args())

    # load the source and reference images
    print("[INFO] loading source and reference images...")
    src = cv2.imread(args["source"])
    ref = cv2.imread(args["reference"])
    # determine if we are performing multichannel histogram matching
    # and then perform histogram matching itself
    print("[INFO] performing histogram matching...")
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel=multi)
    # show the output images
    # cv2.imshow("Source", src)
    # cv2.imshow("Reference", ref)
    # cv2.imshow("Matched", matched)
    # cv2.waitKey(0)

    cv2.imwrite(args["save_filename"], matched)

    # construct a figure to display the histogram plots for each channel
    # before and after histogram matching was applied
    (fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    # loop over our source image, reference image, and output matched
    # image
    for (i, image) in enumerate((src, ref, matched)):
        # convert the image from BGR to RGB channel ordering
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # loop over the names of the channels in RGB order
        for (j, color) in enumerate(("red", "green", "blue")):
            # compute a histogram for the current channel and plot it
            (hist, bins) = exposure.histogram(image[..., j],
                                              source_range="dtype")
            axs[j, i].plot(bins, hist / hist.max())
            # compute the cumulative distribution function for the
            # current channel and plot it
            (cdf, bins) = exposure.cumulative_distribution(image[..., j])
            axs[j, i].plot(bins, cdf)
            # set the y-axis label of the current plot to be the name
            # of the current color channel
            axs[j, 0].set_ylabel(color)

    # set the axes titles
    axs[0, 0].set_title("Source")
    axs[0, 1].set_title("Reference")
    axs[0, 2].set_title("Matched")
    # display the output plots
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if False:
        src = cv2.imread('E:/Downloads/BData/images_with_alpha/im3_blur5.png', cv2.IMREAD_UNCHANGED)
        # ref = cv2.imread('E:/Downloads/BData/images_with_alpha/031.png', cv2.IMREAD_UNCHANGED)
        ref = cv2.imread('E:/Downloads/BData/01/007.JPG', cv2.IMREAD_UNCHANGED)
        print(ref.shape)
        if ref.shape[2] == 4:
            alpha = ref[:,:,3]
            print(np.min(alpha), np.max(alpha))
            print(np.where(alpha))
            x,y = np.where(alpha > 0)
            xmin,ymin,xmax,ymax = np.min(x),np.min(y),np.max(x),np.max(y)
            ref = ref[xmin:xmax, ymin:ymax,:]
            print(ref.shape)
        multi = True if src.shape[-1] > 1 else False
        matched = exposure.match_histograms(src[:,:,:3], ref[:,:,:3], multichannel=multi)
        cv2.imwrite('E:/Downloads/BData/images_with_alpha/im3_blur5_031.png',
                    np.concatenate([matched, src[:,:,3:]], axis=2))
    else:
        src = cv2.imread('E:/Downloads/BData/01/001.JPG', cv2.IMREAD_UNCHANGED)
        # ref = cv2.imread('E:/Downloads/BData/images_with_alpha/031.png', cv2.IMREAD_UNCHANGED)
        ref = cv2.imread('E:/Downloads/BData/01/031.JPG', cv2.IMREAD_UNCHANGED)
        print(ref.shape)
        if ref.shape[2] == 4:
            alpha = ref[:, :, 3]
            print(np.min(alpha), np.max(alpha))
            print(np.where(alpha))
            x, y = np.where(alpha > 0)
            xmin, ymin, xmax, ymax = np.min(x), np.min(y), np.max(x), np.max(y)
            ref = ref[xmin:xmax, ymin:ymax, :]
            print(ref.shape)
        multi = True if src.shape[-1] > 1 else False
        matched = exposure.match_histograms(src[:, :, :3], ref[:, :, :3], multichannel=multi)
        cv2.imwrite('E:/Downloads/BData/images_with_alpha/001_031.png',
                    matched)