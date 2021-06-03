import os, glob

def work(data_root):
    files = glob.glob(data_root+'/*.tif')
    for f in files:
        os.system('gdalinfo %s' % f)


work(r'G:\gddata\aerial')
work(r'G:\gddata\satellite')