from gdal import ogr
# https://gis.stackexchange.com/questions/217165/how-to-create-polygon-shapefile-from-a-list-of-coordinates-using-python-gdal-ogr

def create_polygon(coords):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()

def write_shapefile(poly, out_shp):
    """
    https://gis.stackexchange.com/a/52708/8104
    """
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(out_shp)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    ## If there are multiple geometries, put the "for" loop here

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkt(poly)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None

def main(coords, out_shp):
    poly = create_polygon(coords)
    write_shapefile(poly, out_shp)


def test2():
    # https://gis.stackexchange.com/questions/354016/create-an-esri-shapefile-from-wkt-with-ogr
    import osgeo.osr as osr
    import csv
    import os
    os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
    os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'

    myInput = 'input.csv'

    with open(myInput, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';', quotechar='"')

        # set up the shapefile driver
        driver = ogr.GetDriverByName("ESRI Shapefile")

        # create the data source
        data_source = driver.CreateDataSource("output.shp")

        # create the spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(2154)

        # create the layer
        layer = data_source.CreateLayer("output", srs, ogr.wkbPolygon)

        # Add the fields we're interested in
        field_serialnumber = ogr.FieldDefn("serialnumber", ogr.OFTString)
        field_serialnumber.SetWidth(9)
        layer.CreateField(field_serialnumber)
        field_name = ogr.FieldDefn("name", ogr.OFTString)
        field_name.SetWidth(50)
        layer.CreateField(field_name)
        layer.CreateField(ogr.FieldDefn("objectnumber", ogr.OFTInteger))
        field_city = ogr.FieldDefn("city", ogr.OFTString)
        field_city.SetWidth(5)
        layer.CreateField(field_city)

        # Process the text file and add the attributes and features to the shapefile
        for row in reader:
            # create the feature
            feature = ogr.Feature(layer.GetLayerDefn())
            # Set the attributes using the values from the delimited text file
            feature.SetField("serialnumber", row['serialnumber'])
            feature.SetField("name", row['name'])
            feature.SetField("objectnumber", row['objectnumber'])
            feature.SetField("city", row['city'])

            # create the WKT for the feature using Python string formatting
            wkt = (row['geom'])
            polygon = ogr.CreateGeometryFromWkt(wkt)

            # Set the feature geometry using the point
            feature.SetGeometry(polygon)

            # Create the feature in the layer (shapefile)
            layer.CreateFeature(feature)

            # Dereference the feature
            feature = None

        # Save and close the data source
        data_source = None

if __name__ == "__main__":
    coords = [(-106.6472953, 24.0370137),
              (-106.4933356, 24.05293569),
              (-106.4941789, 24.01969175),
              (-106.4927777, 23.98804445),
              (-106.4922614, 23.95582128),
              (-106.4925834, 23.92302327),
              (-106.4924068, 23.89048039),
              (-106.4925695, 23.85771361),
              (-106.4932479, 23.82457675),
              (-106.4928676, 23.7922049),
              (-106.4925072, 23.75980241),
              (-106.492388, 23.72722475),
              (-106.4922574, 23.69464296),
              (-106.4921181, 23.6620529),
              (-106.4922734, 23.62926733),
              (-106.4917201, 23.59697561),
              (-106.4914134, 23.56449628),
              (-106.4912558, 23.5319045),
              (-106.491146, 23.49926362),
              (-106.4911734, 23.46653561),
              (-106.4910181, 23.43392476),
              (-106.4910156, 23.40119976),
              (-106.4909501, 23.3685223),
              (-106.4908165, 23.33586566),
              (-106.4907735, 23.30314904),
              (-106.4906954, 23.27044931),
              (-106.4906366, 23.23771759),
              (-106.4905894, 23.20499124),
              (-106.4905432, 23.17226022),
              (-106.4904748, 23.1395177),
              (-106.4904187, 23.10676788), (-106.4903676, 23.07401321), (-106.4903098, 23.04126832), (-106.4902512, 23.00849426), (-106.4901979, 22.97572025), (-106.490196, 22.97401001), (-106.6481193, 22.95609832), (-106.6481156, 22.95801668), (-106.6480697, 22.99082052), (-106.6480307, 23.02362441), (-106.6479937, 23.0563977), (-106.6479473, 23.0891833), (-106.647902, 23.12196713), (-106.6478733, 23.15474057), (-106.6478237, 23.18750353), (-106.6477752, 23.22026138), (-106.6477389, 23.25302505), (-106.647701, 23.28577123), (-106.6476562, 23.31851549), (-106.6476211, 23.3512557), (-106.6475745, 23.38397935), (-106.6475231, 23.41671055), (-106.6474863, 23.44942382), (-106.6474432, 23.48213255), (-106.6474017, 23.51484861), (-106.6474626, 23.54747418), (-106.647766, 23.57991134), (-106.6482374, 23.61220905), (-106.6484783, 23.64467084), (-106.6482308, 23.6775148), (-106.6479338, 23.7103854), (-106.6478395, 23.74309074), (-106.6472376, 23.77618646), (-106.6472982, 23.80876072), (-106.647127, 23.84151129), (-106.6471277, 23.8741312), (-106.6473995, 23.90656505), (-106.6473138, 23.93916488), (-106.6473408, 23.97172031), (-106.6473796, 24.00435261), (-106.6472953, 24.0370137)]
    out_shp = r'X:\temp\polygon.shp'
    main(coords, out_shp)

__autor__ = """Igor Lugo (lugoigor@gmail.com)"""

import sys, os
import numpy as np
from osgeo import ogr

# Importing data
openFileLines = 'path/to/lineString/data/as/dictionary.npy'
length_lineString = np.load(openFileLines).item()


def createShapefile():
    # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Setting work directory
    os.chdir('path/to/save/new/shapefile/dir/name')

    # Creating a new data source and layer
    if os.path.exists('shapefileName.shp'):
        driver.DeleteDataSource('shapefileName.shp')

    ds = driver.CreateDataSource('shapefileName.shp')
    if ds is None:
        print ('Could not create file')
        sys.exit(1)

    layer = ds.CreateLayer('layerName', geom_type=ogr.wkbLineString)

    # add a field to the output
    fieldDefn = ogr.FieldDefn('fieldName', ogr.OFTReal)
    layer.CreateField(fieldDefn)

    cnt = 0

    for k, v in length_lineString.iteritems():
        cnt += 1
        lineString = ogr.Geometry(ogr.wkbLineString)
        for m in v:
            lineString.AddPoint(m[0], m[1])

        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(lineString)
        feature.SetField('fieldName', k)
        layer.CreateFeature(feature)

        lineString.Destroy()
        feature.Destroy()

    ds.Destroy()

    print ("Shapefile created")