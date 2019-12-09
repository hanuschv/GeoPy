########################################################################################################################
#
# Assignment 8 - Vector processing II
# Vincent Hanuschik (code from actual group: Lingzhi Zhang & Leon Nill)
#
########################################################################################################################

# ======================================================================================================================
# IMPORT & DEFINE
# ======================================================================================================================
import time
import ogr
import osr
import numpy as np
import os
import itertools
from tqdm import tqdm
import subprocess

time_start = time.localtime()
time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_start)
print("--------------------------------------------------------")
print("Start: " + time_start_str)
print("--------------------------------------------------------")

# ======================================================================================================================
# FUNCTIONS
# ======================================================================================================================
def sample_random(target_geom, n_samples=50, within=True, buffer=None, grid_ref=None, grid_size=None, min_distance=None):
    """
    Stratified random sampling of n-samples for a target geometry.
    :param target_geom: Geometry to sample in.
    :param n_samples: Number of samples.
    :param within: Check wether ti should lie within the polygon (True) or just the bounding box of the target geometry.
    :param grid_ref: (list) List containing x and y coordinates of a reference point, e.g. [x, y]
    :param grid_size: (int/flt) Grid size / pixel size of reference code.
    :param min_distance: (int/flt) Minimum distance between sample points.
    :return:
    """
    # Initialise values
    samples = []
    xmin, xmax, ymin, ymax = target_geom.GetEnvelope()
    distance_logical = False

    if grid_ref:
        x_ref = grid_ref[0]
        y_ref = grid_ref[1]

        xmin = x_ref + (int((xmin - x_ref) / grid_size)) * grid_size
        xmax = x_ref + (int((xmax - x_ref) / grid_size)) * grid_size
        ymin = y_ref + (int((ymin - y_ref) / grid_size)) * grid_size
        ymax = y_ref + (int((ymax - y_ref) / grid_size)) * grid_size

    while len(samples) < n_samples:
        sample_x = np.random.choice(np.arange(xmin, xmax, grid_size))
        sample_y = np.random.choice(np.arange(ymin, ymax, grid_size))

        if min_distance & len(samples) > 0:
            distance = abs(np.subtract(samples, (sample_x, sample_y)))
            distance_logical = np.any(distance < min_distance)

        if not distance_logical:

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
    order = [4 , 2 , 5 , 8 , 1, 7, 0, 3, 6]         # changes the order of appearance of the items
    uls_sorted = [uls[i] for i in order]            # so that the center pixel is in the first position

    # 2 3 4                3 6 9
    # 5 1 6   instead of   2 5 8
    # 7 8 9                1 4 7

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
os.chdir('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment08_data')

ds_pa = ogr.Open('WDPA_May2018_polygons_GER_select10large.shp')
lyr_pa = ds_pa.GetLayer()

ds_onepoint = ogr.Open('OnePoint.shp')
lyr_onepoint = ds_onepoint.GetLayer()
ext_onepoint = (lyr_onepoint.GetExtent())
ref_grid = (ext_onepoint[0], ext_onepoint[2])

crs_source = lyr_pa.GetSpatialRef()
crs_target = lyr_onepoint.GetSpatialRef()
transform = osr.CoordinateTransformation(crs_source, crs_target)

# Create .shp-file
outDriver = ogr.GetDriverByName('ESRI Shapefile')

# Create the output shp
outDataSource = outDriver.CreateDataSource('/Users/Vince/Desktop/PA_Landsat_3x3_50samples.shp')
crs_target = lyr_onepoint.GetSpatialRef()
outLayer = outDataSource.CreateLayer('test.shp', crs_target, geom_type=ogr.wkbPolygon)
fieldDef1 = ogr.FieldDefn('pixel', ogr.OFTString)
fieldDef2 = ogr.FieldDefn('pa', ogr.OFTString)
outLayer.CreateField(fieldDef1)
outLayer.CreateField(fieldDef2)

# Get the output Layer's Feature Definition
featureDefn = outLayer.GetLayerDefn()
##################

for i, pa in tqdm(enumerate(lyr_pa)):
    pa_name = pa.GetField('NAME')
    geom = pa.GetGeometryRef()
    geom.Transform(transform)
    points = sample_random(geom, buffer=90, grid_ref=ref_grid, grid_size=30, min_distance=90)
    poly = ogr.Geometry(ogr.wkbPolygon)

    for point in points:
        uls = construct_upper_lefts(point, pixelsize=30, windowsize=3)

        for j, ul in enumerate(uls):
            poly = bounding_box(ul[0], ul[1], pixelsize=30, poly=poly)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outFeature.SetField("pixel", str(str(i+1)+"."+str(j+1)))  # +1 for i and j for humand readable format, e.g. 1.1, 1.2 etc.
            outFeature.SetField("pa", pa_name)
            # Add new feature to output Layer
            outLayer.CreateFeature(outFeature)
            # dereference the feature
            outFeature = None
            poly = None

# close lyr and ds
outLayer = None
outDataSource = None

# create additional KML, runs in terminal but for some reason does not in script on my machine
cmd = 'ogr2ogr -f "KML" /Users/Vince/Desktop/PA_Landsat_3x3_50samples.kml /Users/Vince/Desktop/PA_Landsat_3x3_50samples.shp'
os.system(cmd)
subprocess.call(cmd)
# ======================================================================================================================
# END
# ======================================================================================================================
time_end = time.localtime()
time_end_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_end)
print("--------------------------------------------------------")
print("End: " + time_end_str)
time_diff = (time.mktime(time_end) - time.mktime(time_start)) / 60
hours, seconds = divmod(time_diff * 60, 3600)
minutes, seconds = divmod(seconds, 60)
print("Duration: " + "{:02.0f}:{:02.0f}:{:02.0f}".format(hours, minutes, seconds))
print("--------------------------------------------------------")
