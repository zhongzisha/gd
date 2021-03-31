import tifffile
import cv2
import numpy as np

im = tifffile.imread('E:\gddata\卫星影像（23景-杆塔+导线+树木）\pleiades-0.5m\po008535_gd31_SO19034847-2-01_DS_PHR1B_201906150311529_FR1_PX_E112N23_0909_01728.tif')
print(im.shape)
print(type(im[0,0,0]))
print(im.min(), im.max())