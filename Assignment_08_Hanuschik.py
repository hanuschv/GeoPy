# ==================================================================================================== #
#   Assignment_08                                                                                      #
#   (c) Vincent Hanuschik, 07/12/2019                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import ogr
import osr
import numpy as np
import os
import itertools
from tqdm import tqdm
# ======================================== SET TIME COUNT ============================================ #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# =================================== DATA FILES AND DIRECTORIES====================================== #
os.chdir('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment08_data')

ds_pa = ogr.Open('WDPA_May2018_polygons_GER_select10large.shp')
lyr_pa = ds_pa.GetLayer()

ds_onepoint = ogr.Open('OnePoint.shp')
lyr_onepoint = ds_onepoint.GetLayer()
# ======================================== FUNCTIONS ================================================= #
def sample_random(target_geom, n_samples=50, within=True, buffer=None, grid_ref=None, grid_size=None):
    """

    :param target_geom:
    :param n_samples:
    :param within:
    :param grid_ref:
    :param grid_size:
    :return:
    """
    samples = []
    xmin, xmax, ymin, ymax = target_geom.GetEnvelope()

    if grid_ref:
        x_ref = grid_ref[0]
        y_ref = grid_ref[1]

        xmin_target = x_ref + (int((xmin - x_ref) / grid_size)) * grid_size
        xmax_target = x_ref + (int((xmax - x_ref) / grid_size)) * grid_size
        ymin_target = y_ref + (int((ymin - y_ref) / grid_size)) * grid_size
        ymax_target = y_ref + (int((ymax - y_ref) / grid_size)) * grid_size

        while len(samples) < n_samples:
            sample_x = np.random.choice(np.arange(xmin_target, xmax_target, grid_size))
            sample_y = np.random.choice(np.arange(ymin_target, ymax_target, grid_size))

            if within:
                # construct geometry of random point
                point_geometry = ogr.Geometry(ogr.wkbPoint)
                point_geometry.AddPoint(sample_x, sample_y)

                if buffer:
                    point_geometry = point_geometry.Buffer(buffer)

                if point_geometry.Within(target_geom):
                    samples.append((sample_x, sample_y))

            else:
                samples.append((sample_x, sample_y))

    else:
        while len(samples) < n_samples:
            sample_x = np.random.choice(np.arange(xmin, xmax, grid_size))
            sample_y = np.random.choice(np.arange(ymin, ymax, grid_size))
            samples.append((sample_x, sample_y))

            if within:
                # construct geometry of random point
                point_geometry = ogr.Geometry(ogr.wkbPoint)
                point_geometry.AddPoint(sample_x, sample_y)

                if buffer:
                    point_geometry = point_geometry.Buffer(buffer)

                if point_geometry.Within(target_geom):
                    samples.append((sample_x, sample_y))

            else:
                samples.append((sample_x, sample_y))

    return samples

def construct_upper_lefts(point, pixelsize, windowsize):
    """

    :param point:
    :param pixelsize:
    :param windowsize:
    :return:
    """
    add_factor = (windowsize / 2) * pixelsize
    ulx, uly = (point[0]-add_factor, point[1]+add_factor)
    ulx_max = ulx+(windowsize*pixelsize)
    uly_max = uly+(windowsize*pixelsize)

    xs = np.arange(ulx, ulx_max, pixelsize)
    ys = np.arange(uly, uly_max, pixelsize)

    uls = list(itertools.product(*[xs, ys]))
    order = [4 , 2 , 5 , 8 , 1, 7, 0, 3, 6]
    uls_sorted = [uls[i] for i in order]

    return uls_sorted

def bounding_box(ulx, uly, pixelsize=30, poly=None):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ulx, uly)
    ring.AddPoint(ulx+pixelsize, uly)
    ring.AddPoint(ulx+pixelsize, uly-pixelsize)
    ring.AddPoint(ulx, uly-pixelsize)
    ring.AddPoint(ulx, uly)

    if poly:
        poly.AddGeometry(ring)
    else:
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

    return poly

# ======================================================================================================================
# EXECUTION
# ======================================================================================================================

ext_onepoint = (lyr_onepoint.GetExtent())
ref_grid = (ext_onepoint[0], ext_onepoint[2])

crs_source = lyr_pa.GetSpatialRef()
crs_target = lyr_onepoint.GetSpatialRef()
transform = osr.CoordinateTransformation(crs_source, crs_target)

# Create .shp-file
outDriver = ogr.GetDriverByName('ESRI Shapefile')

# Create the output shp
outDataSource = outDriver.CreateDataSource('sample_polygons.shp')
crs_target = lyr_onepoint.GetSpatialRef()
outLayer = outDataSource.CreateLayer('test.shp', crs_target, geom_type=ogr.wkbPolygon)
fieldDef1 = ogr.FieldDefn('identifier', ogr.OFTString)
fieldDef2 = ogr.FieldDefn('pa', ogr.OFTString)
outLayer.CreateField(fieldDef1)
outLayer.CreateField(fieldDef2)

# Get the output Layer's Feature Definition
featureDefn = outLayer.GetLayerDefn()
##################

for pa in tqdm(lyr_pa):

    pa_name = pa.GetField('NAME')
    geom = pa.GetGeometryRef()
    geom.Transform(transform)

    points = sample_random(geom, buffer=60, grid_ref=ref_grid, grid_size=30)
    poly = ogr.Geometry(ogr.wkbPolygon)

    for i, point in enumerate(points):
        uls = construct_upper_lefts(point, pixelsize=30, windowsize=3)
        for j, ul in enumerate(uls):
            poly = bounding_box(ul[0], ul[1], pixelsize=30, poly=poly)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outFeature.SetField("identifier",  float(str(str(i+1)+'.'+str(j+1))))
            outFeature.SetField("pa", pa_name)
            # Add new feature to output Layer
            outLayer.CreateFeature(outFeature)
            # dereference the feature
            outFeature = poly = None

# close lyr and ds
outLayer = None
outDataSource = None



# =============================== END TIME-COUNT AND PRINT TIME STATS ============================== #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")