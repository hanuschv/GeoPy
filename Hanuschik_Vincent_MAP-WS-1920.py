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
from tqdm import tqdm
from matplotlib import pyplot as plt
# ======================================== SET TIME COUNT ============================================ #
time_start = time.localtime()
time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_start)
print("--------------------------------------------------------")
print("Start: " + time_start_str)
print("--------------------------------------------------------")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
dir = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/MAP/Week15 - MAP/Datasets/'
os.chdir(dir)
# Create output directory
try :
    outdir = dir+'HANUSCHIK_VINCENT_MAP-WS- 1920_RF-predictions/'
    os.mkdir(outdir)
    print("Directory " , outdir , " Created ")
except FileExistsError :
    print("Directory " , outdir , " already exists")
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
                                                         'Band_12' : landsat_values[11]}, ignore_index=True)
    pointlyr.SetSpatialFilter(None)
    tile = None
print(tc_landsat_metrics[0:5])

# ------------------------ Part 1:  Excersise 3)  --------------------------------#
y = tc_landsat_metrics['tree_fraction']
X = tc_landsat_metrics.drop(['ID', 'tree_fraction'], axis =1)

from sklearn.ensemble import RandomForestRegressor
tree_fraction_model = RandomForestRegressor(n_estimators=500, random_state=0, oob_score=True)
tree_fraction_model.fit(X, y)
# GridSearch_CV -> n_trees & n_splits

# --------- Part 1:  Excersise 3.1) & 3.2)  ----------------#
# 1) the root-mean-squared error of the out-of-bag predicted tree cover versus the observed tree cover,
# 2) the coefficient of determination R^2.

from sklearn import metrics
oob_score = tree_fraction_model.oob_score_
oob_pred = tree_fraction_model.oob_prediction_

mse = metrics.mean_squared_error(oob_pred, y)
rmse = math.sqrt(mse)/10000
r2 = tree_fraction_model.score(X,y)

tile1 = gdal.Open(rasterfiles[0]).ReadAsArray()

# --------- Part 1:  Excersise 4)  ----------------#
# Parallel: Write helper function to flatten tile and apply prediction function

def parallel_predict(list):
    model = list[1]
    # scaler = list[2]
    raster = gdal.Open(list[0])
    img = raster.ReadAsArray()

    ydim = img.shape[1]
    xdim = img.shape[2]
    tile = img.transpose(1 , 2 , 0).reshape((ydim * xdim , img.shape[0]))

    predict = model.predict(tile[:,0:12])
    rs = predict.reshape((ydim , xdim))
    outPath = outdir + os.path.basename(list[0])
    outPath = outPath.replace(".tif" , "_RF-regression.tif")
    array2geotiff(rs,outPath, ingdal=raster)
    gdal_pyramids = gdal.Open(outPath)
    gdal_pyramids.BuildOverviews('average', [2, 4, 8, 16, 32])
    gdal_pyramids = None
    return rs

pred_arg_list = [(raster, tree_fraction_model) for raster in rasterfiles]

from joblib import Parallel, delayed
output = Parallel(n_jobs=3)(delayed(parallel_predict)(list) for list in pred_arg_list)

# pred_rast_list =
# gdal.buildvrt doq_index.vrt doq/*.tif


# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
# for raster in rasterfiles[0:1]:
#     # open tile
#     tile = gdal.Open(raster).ReadAsArray()
#     n_bands, n_rows, n_cols = tile.shape
#     new_shape = (n_rows * n_cols, n_bands)
#     flatTile = tile.reshape((n_bands, n_rows * n_cols))
#     flatTile.shape
#     flatTile = np.transpose(flatTile)
#     print('Reshaped from {o} to {n}'.format(o=tile.shape ,
#                                             n=flatTile.shape))
#
#     # Now predict for each pixel
#     class_prediction = tree_fraction_model.predict(flatTile[:,0:12])
#     # Reshape our classification map
#     tree_fraction_tile = class_prediction.reshape(tile[0, :, :].shape)
#     tile_name = os.path.basename(raster)
#     array2geotiff(tree_fraction_tile , tile_name+'_RF-regression.tif' , ingdal=gdal.Open(raster))
#     tile = None


# array2geotiff(tree_fraction_tile, 'tree_fraction_tile01.tif', ingdal= gdal.Open(rasterfiles[0]))
plt.imshow(tree_fraction_tile)
plt.show()

mosaic_files = ListFiles(outdir, '.tif',1)
mosaic_tiles = [gdal.Open(tile) for tile in mosaic_files]

# vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
vrt = gdal.BuildVRT(outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.vrt', mosaic_files)
vrt = None   # necessary in order to write to disk

vrt_to_tiff = gdal.Open(outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.vrt')
vrt_arr = vrt_to_tiff.ReadAsArray()
# vrt_arr[vrt_arr == 0 ] = np.nan

array2geotiff(vrt_arr, outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.tif', ingdal=vrt_to_tiff)
tiff_overview = gdal.Open(outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.tif', 0)
tiff_overview = tiff_overview.BuildOverviews('average', [2, 4, 8, 16, 32])
tiff_overview = None
# gdal.Open(. , 0) opens it read-only -> .ovr's created externally, ',1' stores them internally (read-write)

vrt_to_tiff = gdal.Open(outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.vrt')
vrt_to_tiff = vrt_to_tiff.BuildOverviews('average', [2, 4, 8, 16, 32])
vrt_ov = None

# gdalmerge, createcopy
# pyramid layer level all (2-32) use gdaladdo
# copy raster to new raster

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part II ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
polyfiles = ListFiles(polydir, '.shp', 1)
polyshp = ogr.Open(polydir+'FL_Areas_ChacoRegion.shp')
polylyr = polyshp.GetLayer()
tree_fraction_mosaic = gdal.Open(outdir +'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.tif')

gt = tree_fraction_mosaic.GetGeoTransform()
pixel_size = gt[1]

crs_tree_fraction = osr.SpatialReference(wkt=tree_fraction_mosaic.GetProjection())
crs_poly = polylyr.GetSpatialRef()
transform_poly = osr.CoordinateTransformation(crs_poly,crs_tree_fraction)
outDriver = ogr.GetDriverByName("Memory")

mosaic_extent = create_poly_extent(tree_fraction_mosaic)

def image_offsets(img, coordinates):
    gt = img.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    offsets_ul = list(map(int, gdal.ApplyGeoTransform(inv_gt, coordinates[0], coordinates[1])))
    offsets_lr = list(map(int, gdal.ApplyGeoTransform(inv_gt, coordinates[2], coordinates[3])))
    return offsets_ul + offsets_lr

# plylyr_names = [field.name for field in polylyr.schema]

summary = pd.DataFrame(columns={'ID': [],
                                'OT_class': [],
                                'area_km2': [],
                                'mean': [],
                                'standard_deviation': [],
                                'min': [],
                                'max': [],
                                '10th_percentile': [],
                                '90th percentile': []})


polylyr.ResetReading()
within = []
out =[]
for poly in polylyr:
    # ------------ mean elevation of parcel ---------
    # Geometry of each parcel
    ID = poly.GetField('UniqueID')
    OT_class = poly.GetField('OT_class')
    geom = poly.GetGeometryRef().Clone()
    degarea = geom.GetArea()
    geom.Transform(transform_poly)
    if geom.Within(mosaic_extent):
        #something
    else:
        #something

    Envelope = geom.GetEnvelope()
    x_min , x_max , y_min , y_max = geom.GetEnvelope()

    # create temporary shape file with only the current feature
    outDataSource = outDriver.CreateDataSource('temp.shp')
    outLayer = outDataSource.CreateLayer('' , crs_tree_fraction , geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    # create feature
    poly = ogr.Feature(featureDefn)
    poly.SetGeometry(geom)
    outLayer.CreateFeature(poly)

    # Get resolution of vectorized raster
    x_res = int((x_max - x_min) / pixel_size)
    # if the parcel is smaller than 1 pixel, 1 pixel will be created
    if x_res == 0 :
        x_res = 1
    y_res = int((y_max - y_min) / pixel_size)
    if y_res == 0 :
        y_res = 1
    # create temporary gdal raster
    target_ds = gdal.GetDriverByName('MEM').Create('' , x_res , y_res , gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min , pixel_size , 0 , y_max , 0 , -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(255)
    # Rasterize parcel
    gdal.RasterizeLayer(target_ds , [1] , outLayer , burn_values=[1])
    poly = None
    outDataSource = None
    outLayer = None
    # read mask
    mask = band.ReadAsArray()
    # get array indices for subsetting the dem
    img_offsets = image_offsets(tree_fraction_mosaic , (x_min , y_max , x_max , y_min))
    # subset dem
    tf_m_sub = tree_fraction_mosaic.ReadAsArray(img_offsets[0] , img_offsets[1] , mask.shape[1] , mask.shape[0])

    # apply mask and calc mean
    dem_mean = np.mean(np.where(mask == 1 , tf_m_sub , 0))

    summary = pd.DataFrame(columns={'ID' : ID ,
                                    'OT_class' : OT_class ,
                                    'area_km2' : degarea,
                                    'mean' : [] ,
                                    'standard_deviation' : [] ,
                                    'min' : [] ,
                                    'max' : [] ,
                                    '10th_percentile' : [] ,
                                    '90th percentile' : []})
    # ------------------------------------------------

polylyr.ResetReading()






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