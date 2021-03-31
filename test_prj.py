from osgeo import gdal, osr, ogr
ds = gdal.Open(r'E:\gddata\aerial2\水口300m_mosaic.tif')
prj = ds.GetProjection()
print(prj)
srs = osr.SpatialReference(wkt=prj)
bb=srs.ExportToWkt(["FORMAT=WKT1_ESRI"])
print(srs)
print(bb)

