import tifffile
import os
import numpy as np
import cv2


# data_root = 'E:\gddata'
# for root, dirs, files in os.walk(data_root, topdown=False):
#     for name in files:
#         filename = os.path.join(root, name)
#         if os.path.isfile(filename) and name[-4:] == '.tif':
#             print(filename)
#             im = tifffile.imread(filename)
#             print(im.shape)
#             del im
#
#     # for name in dirs:
#     #     print(os.path.join(root, name))
import glob
from osgeo import gdal
from natsort import natsorted

subsize = 5120
gap = 512

dsdir = '/media/ubuntu/Working/rs/guangdong_aerial/aerial/'
dstiffiles = natsorted(glob.glob(dsdir+'/*.tif'))
for ii, dsttif in enumerate(dstiffiles):
    print(ii, dsttif)

for ii in range(15):
    tiffiles = natsorted(glob.glob('/media/ubuntu/Temp/gd/data/aerial/%d_%d/%d/%d_*.tif'%(subsize, gap,ii,ii)))
    print(ii)
    # print(tiffiles)

    print(tiffiles[0])

    prefix = tiffiles[0].split(os.sep)[-1].replace('.tif','').split('_')[1:3]
    i, j = int(float(prefix[0])) - 1, int(float(prefix[1])) - 1
    xoffset = j * (subsize - gap)
    yoffset = j * (subsize - gap)

    im1 = cv2.imread(tiffiles[0])[:,:,::-1]
    cv2.imwrite('im1-%d.jpg' % ii, im1)
    height, width = im1.shape[:2]
    for jj, dsttif in enumerate(dstiffiles):
        im2 = np.zeros((height, width, 3), dtype=np.uint8)
        print(jj, dsttif)
        ds = gdal.Open(dsttif, gdal.GA_ReadOnly)
        height2, width2 = ds.RasterYSize, ds.RasterXSize
        if yoffset >= height2 or xoffset >= width2:
            continue
        if (yoffset+subsize) >= height2:
            stepy = height2 - yoffset
        else:
            stepy = subsize
        if (xoffset+subsize) >= width2:
            stepx = width2 - xoffset
        else:
            stepx = subsize
        print(jj, height2, width2)
        for b in range(3):
            data = ds.GetRasterBand(b+1).ReadAsArray(xoffset, yoffset, stepx, stepy)
            im2[:stepy, :stepx, b] = data
            del data
        ds = None

        # check if im1 == im2
        print(ii, jj)

        cv2.imwrite('im2-%d-%d.jpg' % (ii, jj), np.concatenate([im1, im2], axis=1))

        # import pdb
        # pdb.set_trace()





