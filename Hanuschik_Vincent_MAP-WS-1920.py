# ==================================================================================================== #
#   MAP                                                                                      #
#   (c) Vincent Hanuschik, 10/12/2019                                                                  #
#
#
#   # tested and running on a Mac running Python 3.7
# ==================================
# - Metadata on Landsat Spectral Metrics
#      --> Band 1-12!
# - Polygon/Pixel Interaction/overlap --> approach?
# - OT_class meaning?
#
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import math
import gdal
import ogr, osr
import pandas as pd
import numpy as np
import joblib
import multiprocessing
from matplotlib import pyplot as plt
# ======================================== SET TIME COUNT ============================================ #
time_start = time.localtime()
time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_start)
print("--------------------------------------------------------")
print("Start: " + time_start_str)
print("--------------------------------------------------------")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
dir = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/MAP/Week15 - MAP/Datasets/'
# ==================================================================================================== #
pointdir = dir + 'RandomPoint_Dataset/'
rasterdir = dir + 'RasterFiles/'
polydir = dir + 'Zonal_Dataset/'
# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
def ListFiles(filepath , filetype , expression) :
    '''
    # lists all files in a given folder path of a given file extentsion
    :param filepath: string of folder path
    :param filetype: filetype
    :param expression: 1 = gives out the list as file-paths ; 0 = gives out the list as file-names only
    :return: list of files
    :return: list of files
    '''
    file_list = []
    for file in os.listdir(filepath) :
        if file.endswith(filetype) :
            if expression == 0 :
                file_list.append(file)
            elif expression == 1 :
                file_list.append(os.path.join(filepath , file))
    return file_list

def CornerCoordinates(rasterpath):
    '''
    Gets corner coordinates of given raster (filepath). Uses GetGeoTransform to extract coordinate information.
    Returns list with coordinates in [upperleft x, upperleft y, lowerright x and lowerright y] form.
    :param rasterpath:
    :return:
    '''
    raster = gdal.Open(rasterpath)
    gt = raster.GetGeoTransform()  # get geo transform data
    ul_x = gt[0]  # upper left x coordinate
    ul_y = gt[3]  # upper left y coordinate
    lr_x = ul_x + (gt[1] * raster.RasterXSize)  # upper left x coordinate + number of pixels * pixel size
    lr_y = ul_y + (gt[5] * raster.RasterYSize)  # upper left y coordinate + number of pixels * pixel size
    coordinates = [ul_x , ul_y , lr_x , lr_y]
    return coordinates

# def overlapExtent(rasterPathlist):
#     '''
#     Finds the common extent/ overlap and returns geo coordinates from the extent.
#     Returns list with corner coordinates.
#     Uses GetGeoTransform to extract corner values of each raster.
#     Common extent is then calculated by maximum ul_x value,
#     minimum ul_y value, minimum lr_x value and maximum lr_y value.
#     Use list comprehensions to calculate respective coordinates for all rasters and
#     use the index to extract correct position coordinates[upperleft x, upperleft y, lowerright x and lowerright y].
#     :param rasterPathlist:
#     :return:
#     '''
#     ul_x_list = [CornerCoordinates(path)[0] for path in rasterPathlist]
#     ul_y_list = [CornerCoordinates(path)[1] for path in rasterPathlist]
#     lr_x_list = [CornerCoordinates(path)[2] for path in rasterPathlist]
#     lr_y_list = [CornerCoordinates(path)[3] for path in rasterPathlist]
#     overlap_extent = []
#     overlap_extent.append(max(ul_x_list))
#     overlap_extent.append(min(ul_y_list))
#     overlap_extent.append(min(lr_x_list))
#     overlap_extent.append(max(lr_y_list))
#     return overlap_extent

def create_poly_extent(img):
    gt = img.GetGeoTransform()

    ulx = gt[0]
    uly = gt[3]
    lrx = gt[0] + gt[1] * img.RasterXSize
    lry = gt[3] + gt[5] * img.RasterYSize
    ext = [ulx, uly, lrx, lry]

    ring = ogr.Geometry(ogr.wkbLinearRing)
    poly = ogr.Geometry(ogr.wkbPolygon)

    ring.AddPoint(ext[0], ext[1])
    ring.AddPoint(ext[2], ext[1])
    ring.AddPoint(ext[2], ext[3])
    ring.AddPoint(ext[0], ext[3])
    ring.AddPoint(ext[0], ext[1])

    poly.AddGeometry(ring)

    return poly

def array2geotiff(inputarray, outputfilename, gt = None, pj = None, ingdal = None, dtype = gdal.GDT_Float32):
    if len(inputarray.shape) > 2:
        nrow = inputarray.shape[1]
        ncol = inputarray.shape[2]
        zdim = inputarray.shape[0]
    else:
        nrow = inputarray.shape[0]
        ncol = inputarray.shape[1]
        zdim = 1
    if gdal is not None:
        gt = ingdal.GetGeoTransform()
        pj = ingdal.GetProjection()

    drv = gdal.GetDriverByName('GTiff')
    # dtype = np2gdal_datatype[str(inputarray.dtype)]
    ds = drv.Create(outputfilename, ncol , nrow , zdim, dtype)
    ds.SetGeoTransform(gt)
    ds.SetProjection(pj)
    if len(inputarray.shape) > 2:
        for i in range(zdim):
            zarray = inputarray[i, :, :]
            ds.GetRasterBand(i+1).WriteArray(zarray)
    else:
        ds.GetRasterBand(1).WriteArray(inputarray)
    ds.FlushCache()
# =======================================  Part 1 ==================================================== #
pointfiles = ListFiles(pointdir, '.shp', 1)
pointshp = ogr.Open(pointfiles[0])

rasterfiles = sorted(ListFiles(rasterdir, '.tif', 1))
# rastertiles = [gdal.Open(tile) for tile in rasterfiles]
# rasterarr = [gdal.Open(tile).ReadAsArray() for tile in rasterfiles]

point_lyr = pointshp.GetLayer()
pointlyr_names = [field.name for field in point_lyr.schema]

# get Projections of Layers to check coordinate systems. In this case both are projected using EPSG 4326/WGS84
pointcrs = point_lyr.GetSpatialRef()
rastercrs = gdal.Open(rasterfiles[0]).GetProjection()
# polygoncrs = poly_lyr.GetSpatialRef()
# necessary to do tasks individually?
'''
tile0 = gdal.Open(rasterfiles[0])
extent = create_poly_extent(tile0)
lyr_pt = pointshp.GetLayer()
lyr_pt.SetSpatialFilter(extent)
'''

# extract response variable
# tc_values = pd.DataFrame(columns={'ID': [],
#                                 'tree_fraction': []
#                                 })
# for point in point_lyr:
#     ID = point.GetField('CID')
#     tree_fraction = point.GetField('TC')
#     # handle NA values
#     if not tree_fraction == -9999:
#     tc_values = tc_values.append({'ID': ID,
#                                   'tree_fraction': tree_fraction
#                                   }, ignore_index=True)
# point_lyr.ResetReading()

# ------------------------ Part 1:  Excersise 1) & 2)  --------------------------------#

tc_landsat_metrics = pd.DataFrame(columns={'ID' : [] ,
                                        'tree_fraction' : [],
                                        'Band_01': [],
                                        'Band_02': [],
                                        'Band_03': [],
                                        'Band_04': [],
                                        'Band_05': [],
                                        'Band_06': [],
                                        'Band_07': [],
                                        'Band_08': [],
                                        'Band_09': [],
                                        'Band_10': [],
                                        'Band_11': [],
                                        'Band_12': [],
                                       })
for raster in rasterfiles:
    # open tile
    tile = gdal.Open(raster)
    gt = tile.GetGeoTransform()
    # Get Extent from raster tile and create Polygon extent
    extent = create_poly_extent(tile)
    # Select points within tile using SpatialFilter()
    pointlyr = pointshp.GetLayer()
    pointlyr.SetSpatialFilter(extent)
    # tile_values = [] # create list for each tile, where values of point response variable is stored
    # tile_arr = tile.ReadAsArray()
    for point in pointlyr:
        ID = point.GetField('CID')
        tree_fraction = point.GetField('TC')
        if not tree_fraction == -9999 :
            geom = point.GetGeometryRef().Clone()
            mx, my = geom.GetX(), geom.GetY()
            px = int((mx - gt[0]) / gt[1])  # x pixel
            py = int((my - gt[3]) / gt[5])  # y pixel

            landsat_values = tile.ReadAsArray(px, py, 1, 1).flatten()

            tc_landsat_metrics = tc_landsat_metrics.append({'ID' : ID ,
                                                         'tree_fraction' : tree_fraction ,
                                                         'Band_01' : landsat_values[0],
                                                         'Band_02' : landsat_values[1],
                                                         'Band_03' : landsat_values[2],
                                                         'Band_04' : landsat_values[3],
                                                         'Band_05' : landsat_values[4],
                                                         'Band_06' : landsat_values[5],
                                                         'Band_07' : landsat_values[6],
                                                         'Band_08' : landsat_values[7],
                                                         'Band_09' : landsat_values[8],
                                                         'Band_10' : landsat_values[9],
                                                         'Band_11' : landsat_values[10],
                                                         'Band_12' : landsat_values[11],}, ignore_index=True)
    pointlyr.SetSpatialFilter(None)
    tile = None
print(tc_landsat_metrics[0:5])

# ------------------------ Part 1:  Excersise 3)  --------------------------------#
y = tc_landsat_metrics['tree_fraction']
X = tc_landsat_metrics.drop(['ID', 'tree_fraction'], axis =1)

from sklearn.ensemble import RandomForestRegressor
tree_fraction_model = RandomForestRegressor(n_estimators=500, random_state=0, oob_score=True, n_jobs=3)
tree_fraction_model.fit(X, y)

# --------- Part 1:  Excersise 3.1) & 3.2)  ----------------#
# 1) the root-mean-squared error of the out-of-bag predicted tree cover versus the observed tree cover,
# 2) the coefficient of determination R^2.

from sklearn import metrics
oob_score = tree_fraction_model.oob_score_
oob_pred = tree_fraction_model.oob_prediction_

mse = metrics.mean_squared_error(oob_pred, y)
rmse = math.sqrt(mse)/10000
r2 = tree_fraction_model.score(X,y)


# --------- Part 1:  Excersise 4)  ----------------#
# Parallel: Write helper function to flatten tile and apply prediction function
def parallel_predict(list):
    model = list[1]
    # scaler = list[2]
    raster = gdal.Open(list[0])
    img = raster.ReadAsArray()

    ydim = img.shape[1]
    xdim = img.shape[2]
    landsat = img.transpose(1 , 2 , 0).reshape((ydim * xdim , img.shape[0]))
    landsat_z = scaler.fit_transform(landsat)

    classification = model.predict(landsat_z)
    rs = classification.reshape((ydim , xdim))
    outPath = list[0]
    outPath = outPath.replace(".tif" , "_RF-regression.tif")
    array2geotiff(rs,outPath, ingdal=raster)
    return rs
# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
for raster in rasterfiles:
    # open tile
    tile = gdal.Open(raster).ReadAsArray()
    n_bands, n_rows, n_cols = tile.shape
    new_shape = (n_rows * n_cols, n_bands)
    flatTile = tile.reshape((n_bands, n_rows * n_cols))
    flatTile.shape
    flatTile = np.transpose(flatTile)
    print('Reshaped from {o} to {n}'.format(o=tile.shape ,
                                            n=flatTile.shape))

    # Now predict for each pixel
    class_prediction = tree_fraction_model.predict(flatTile)
    # Reshape our classification map
    tree_fraction_tile = class_prediction.reshape(tile[0, :, :].shape)
    tile = None


# array2geotiff(tree_fraction_tile, 'tree_fraction_tile01.tif', ingdal= gdal.Open(rasterfiles[0]))
plt.imshow(tree_fraction_tile)
plt.show()

# gdal.BuildVRT
# gdalmerge, createcopy
# pyramid layer level all (2-32) use gdaladdo
# copy raster to new raster
# Create directory
# dirName = 'output'
#
# try :
#     # Create target Directory
#     os.mkdir(dirName)
#     print("Directory " , dirName , " Created ")
# except FileExistsError :
#     print("Directory " , dirName , " already exists")




# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise XX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #

# =============================== END TIME-COUNT AND PRINT TIME STATS ============================== #
time_end = time.localtime()
time_end_str = time.strftime("%a, %d %b %Y %H:%M:%S" , time_end)
print("--------------------------------------------------------")
print("End: " + time_end_str)
time_diff = (time.mktime(time_end) - time.mktime(time_start)) / 60
hours , seconds = divmod(time_diff * 60 , 3600)
minutes , seconds = divmod(seconds , 60)
print("Duration: " + "{:02.0f}:{:02.0f}:{:02.0f}".format(hours , minutes , seconds))
print("--------------------------------------------------------")