# ==================================================================================================== #
#   MAP  - Geoprocessing with Python                                                                   #
#   (c) Vincent Hanuschik, 10/12/2019                                                                  #
#                                                                                                      #
#   # tested and running on a Mac (Mac OS Mojave 10.14.6), Python 3.7 and Gdal 2.3.3                   #
#                                                                                                      #
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

# print gdal version (2030300 in my case, meaning 2.3.3)
print("Gdal Version is:", gdal.VersionInfo())

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
    print("Directory " , outdir , " successfully Created ")
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

def create_poly_extent(img):
    '''
    Returns a polygon from a gdal Raster
    :param img:
    :return:
    '''
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

def array2geotiff(inputarray, outputfilename, pyramids=False, gt = None, pj = None, ingdal = None, dtype = gdal.GDT_Float32):
    '''
    Writes a given array as a .tif file to disk. Optionally creates pyramids/overviews. Can use a gdal raster as input
    for coordinate system and projection information.
    :param inputarray:
    :param outputfilename:
    :param pyramids:
    :param gt:
    :param pj:
    :param ingdal:
    :param dtype:
    :return:
    '''
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
    if pyramids == True:
        gdal_pyramids = gdal.Open(outputfilename)
        gdal_pyramids.BuildOverviews('average', [2, 4, 8, 16, 32])
        gdal_pyramids = None

def parallel_predict(list):
    '''
    helper function to flatten tile and apply prediction function and build pyramids for prediction
    :param list: list[0] raster path string ; list[1] random forest model used in prediction
    :return:
    '''
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
    array2geotiff(rs,outPath,pyramids = True, ingdal=raster)
    print(os.path.basename(outPath), "successfully created.")
    return rs

def image_offsets(img, coordinates):
    '''
    Function that calculates the image offsets from the upper left and lower right corners.
    :param img:
    :param coordinates:
    :return:
    '''
    gt = img.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    offsets_ul = list(map(int, gdal.ApplyGeoTransform(inv_gt, coordinates[0], coordinates[1])))
    offsets_lr = list(map(int, gdal.ApplyGeoTransform(inv_gt, coordinates[2], coordinates[3])))
    return offsets_ul + offsets_lr

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part I ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
# load point shapefile and get list with strings of all tiles in rasterfolder
pointfiles = ListFiles(pointdir, '.shp', 1)
pointshp = ogr.Open(pointfiles[0])
point_lyr = pointshp.GetLayer()
pointlyr_names = [field.name for field in point_lyr.schema]

rasterfiles = sorted(ListFiles(rasterdir, '.tif', 1))

# get Projections of Layers to check coordinate systems.
# In this case both rasters and the pointshapefile are projected using EPSG 4326/WGS84.
pointcrs = point_lyr.GetSpatialRef()
rastercrs = gdal.Open(rasterfiles[0]).GetProjection()

# ------------------------ Part 1.1) & 1.2)  --------------------------------#
# create dataFrame with necessary columns (excluding bands 13-21)
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

# for each tile create a polygon in order to filter only points within the tile extent;
# next loop through all filtered points and if they are not NA's (-9999) tansform into
# array coordinates and extract the corresponding values and append to the dataframe.
# reset the filter and start over with the next tile.
print("Looping over tiles, extracting Landsat Metrics and tree fractions.")
for raster in tqdm(rasterfiles):
    # open tile
    tile = gdal.Open(raster)
    gt = tile.GetGeoTransform()
    # Get Extent from raster tile and create Polygon extent
    extent = create_poly_extent(tile)
    # Select points within tile using SpatialFilter()
    pointlyr = pointshp.GetLayer()
    pointlyr.SetSpatialFilter(extent)

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
print("Variable extraction done.")
print(tc_landsat_metrics[0:5])

# ------------------------ Part 1.3)  --------------------------------#
# split the dataframe into explanatory variable (landsat metrics) and response variable (tree fraction)
y = tc_landsat_metrics['tree_fraction']
X = tc_landsat_metrics.drop(['ID', 'tree_fraction'], axis =1)

from sklearn.ensemble import RandomForestRegressor
tree_fraction_model = RandomForestRegressor(n_estimators=200, random_state=0, oob_score=True)
tree_fraction_model.fit(X, y)

# --------- Part 1.3.1) & 1.3.2)  ----------------#
# 1) the root-mean-squared error of the out-of-bag predicted tree cover versus the observed tree cover,
# 2) the coefficient of determination R^2.
from sklearn import metrics
oob_score = tree_fraction_model.oob_score_
oob_pred = tree_fraction_model.oob_prediction_

r2 = metrics.r2_score(y,oob_pred)
rmse = math.sqrt(metrics.mean_squared_error(y, oob_pred))/10000

print("Random Forest Tree Fraction Model created.\n The the root-mean-squared error (RMSE) of the out-of-bag predicted tree cover versus the observed tree cover amounts to:",
      round(rmse,4),".\n The coefficient of determination R^2 results in a score of:", round(r2,4),".")

# --------- Part 1.4)  ----------------#
# predicting each tile in parallel using the helper function defined above
# also includes the calculation of pyramids ( Part 1.7)
pred_arg_list = [(raster, tree_fraction_model) for raster in rasterfiles]

from joblib import Parallel, delayed
print("Starting prediction with Tree Fraction Model on tiles.")
output = Parallel(n_jobs=3)(delayed(parallel_predict)(list) for list in pred_arg_list)

# --------- Part 1.5.1) & Part 1.5.2) ----------------#
# create mosaic for study area and save as .vrt and .tif

# --- 1.5.1) & 1.7)
# build the Virtual Raster File and build Pyramids
mosaic_files = ListFiles(outdir, '.tif',1)
vrt = gdal.BuildVRT(outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.vrt', mosaic_files)
vrt = None   # necessary in order to write to disk
vrt_ovr = gdal.Open(outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.vrt')
vrt_ovr = vrt_ovr.BuildOverviews('average', [2, 4, 8, 16, 32])
vrt_ovr = None
# --- 1.5.2) & 1.7)
# use the mosaic .vrt from above and .ReadAsArray in order to save with array2geotiff
# pyramids are build within the array2geotiff function
vrt_to_tiff = gdal.Open(outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.vrt')
vrt_arr = vrt_to_tiff.ReadAsArray()

array2geotiff(vrt_arr, outdir+'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.tif', pyramids=True, ingdal=vrt_to_tiff)
# gdal.Open(. , 0) opens it read-only -> .ovr's created externally, ',1' stores them internally (read-write)


# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part II ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
# open zonal shapefile and recently created mosaic.tif
polyfiles = ListFiles(polydir, '.shp', 1)
polyshp = ogr.Open(polydir+'FL_Areas_ChacoRegion.shp')
polylyr = polyshp.GetLayer()
tree_fraction_mosaic = gdal.Open(outdir +'HANUSCHIK_VINCENT_MAP-WS-1920_RF-prediction.tif')

# get GeoTransform/pixelsize information needed later
gt = tree_fraction_mosaic.GetGeoTransform()
pixel_size = gt[1]

# The zonal shapefile uses a "South_America_Albers_Equal_Area_Conic" crs and has to be transformed.
polygoncrs = polylyr.GetSpatialRef()

crs_tree_fraction = osr.SpatialReference(wkt=tree_fraction_mosaic.GetProjection())
crs_poly = polylyr.GetSpatialRef()
transform_poly = osr.CoordinateTransformation(crs_poly,crs_tree_fraction)
outDriver = ogr.GetDriverByName("Memory")
mosaic_extent = create_poly_extent(tree_fraction_mosaic)

# create empty dataframe with necessary columns
summary = pd.DataFrame(columns={'ID': [],
                                'OT_class': [],
                                'area_km2': [],
                                'mean': [],
                                'standard_deviation': [],
                                'min': [],
                                'max': [],
                                '10th_percentile': [],
                                '90th percentile': []})

# Start loop over all polygons, get ID, area and class information.
# Since the Polygon is not yet transformed and the unit from the crs is meters,
# the area reported should be in square meters.
# Next the polygon will be checked whether it sits entirely within the mosaic extent or not.
# If true, the polygon feature will be stored in a separate shapefile and then
# rasterized in order to extract the tree fraction and calculate the statistics.
# Lastly they will be appended to the dataframe. If the polygon is not within the mosaic extent
# statistics will be set to np.nan.

polylyr.ResetReading()
print("Calculating zonal tree fraction statistics. Looping over polygon features in zonal shapefile.")
for poly in tqdm(polylyr):
    # ------------ mean elevation of parcel ---------
    # Geometry of each parcel
    ID = poly.GetField('UniqueID')
    OT_class = poly.GetField('OT_class')
    geom = poly.GetGeometryRef().Clone()
    area = geom.GetArea()/1000000
    geom.Transform(transform_poly)

    if geom.Within(mosaic_extent):
        Envelope = geom.GetEnvelope()
        x_min , x_max , y_min , y_max = geom.GetEnvelope()
        # create temporary shape file with only the current feature
        outDataSource = outDriver.CreateDataSource('temp.shp')
        outLayer = outDataSource.CreateLayer('' , crs_tree_fraction , geom_type=ogr.wkbPolygon)
        featureDefn = outLayer.GetLayerDefn()
        # create feature
        poly_ = ogr.Feature(featureDefn)
        poly_.SetGeometry(geom)
        outLayer.CreateFeature(poly_)
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
        # get array indices for subsetting the mosaic
        img_offsets = image_offsets(tree_fraction_mosaic , (x_min , y_max , x_max , y_min))
        # subset dem
        tf_m_sub = tree_fraction_mosaic.ReadAsArray(img_offsets[0] , img_offsets[1] , mask.shape[1] , mask.shape[0])

        # apply mask and calculate statistics
        mean_tree_fraction = np.mean(np.where(mask == 1 , tf_m_sub , 0))
        sd_tree_fraction = np.std(np.where(mask == 1 , tf_m_sub , 0))
        min_tree_fraction = np.min(np.where(mask == 1 , tf_m_sub , 0))
        max_tree_fraction = np.max(np.where(mask == 1 , tf_m_sub , 0))
        tenth_tree_fraction = np.percentile(np.where(mask == 1 , tf_m_sub , 0) , 10)
        ninetieth_tree_fraction = np.percentile(np.where(mask == 1 , tf_m_sub , 0) , 90)

        summary = summary.append({'ID' : ID ,
                                        'OT_class' : OT_class ,
                                        'area_km2' : area ,
                                        'mean' : mean_tree_fraction ,
                                        'standard_deviation' : sd_tree_fraction ,
                                        'min' : min_tree_fraction ,
                                        'max' : max_tree_fraction ,
                                        '10th_percentile' : tenth_tree_fraction ,
                                        '90th percentile' : ninetieth_tree_fraction}, ignore_index=True)
        # ------------------------------------------------
    else:
        summary = summary.append({'ID' : ID ,
                                  'OT_class' : OT_class ,
                                  'area_km2' : area ,
                                  'mean' : "None" ,
                                  'standard_deviation' : "None" ,
                                  'min' : "None" ,
                                  'max' : "None" ,
                                  '10th_percentile' : "None" ,
                                  '90th percentile' : "None"} , ignore_index=True)
polylyr.ResetReading()
print("Finished. Writing Dataframe to disk.")

# lastly the dataframe is written do disk
summary.to_csv(outdir+"HANUSCHIK_VINCENT_MAP-WS-1920_summaryStats.csv",index=False, float_format='%.3f', encoding='utf-8-sig')
print("Script finished. Congratulations.")

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