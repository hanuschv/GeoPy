# ==================================================================================================== #
#   Assignment_09                                                                                      #
#   (c) Vincent Hanuschik, 10/12/2019                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import ogr, osr
import pandas as pd
import ogr
import gdal
from tqdm import tqdm
import lsnrs
import numpy as np

# ======================================== SET TIME COUNT ============================================ #
time_start = time.localtime()
time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_start)
print("--------------------------------------------------------")
print("Start: " + time_start_str)
print("--------------------------------------------------------")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
dir = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment09_data/'

marihuana_grows = ogr.Open(dir + 'Marihuana_Grows.shp')
parcels = ogr.Open(dir + 'Parcels.shp')
public_lands = ogr.Open(dir + 'PublicLands.shp')
roads = ogr.Open(dir + 'Roads.shp')
TimberHarvestPlan = ogr.Open(dir + 'TimberHarvestPlan.shp')
dem = gdal.Open(dir + 'DEM_Humboldt.tif')
# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data preperation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
# load all Layers
parcel_lyr = parcels.GetLayer()
mar_lyr = marihuana_grows.GetLayer()
publiclands_lyr = public_lands.GetLayer()
roads_lyr = roads.GetLayer()
timber_lyr = TimberHarvestPlan.GetLayer()

outDriver = ogr.GetDriverByName("Memory")

gt = dem.GetGeoTransform()
pixel_size = gt[1]
# extract spatial references
crs_dem = osr.SpatialReference(wkt=dem.GetProjection())
crs_grows = mar_lyr.GetSpatialRef()
crs_roads = roads_lyr.GetSpatialRef()
crs_public = publiclands_lyr.GetSpatialRef()
crs_timber = timber_lyr.GetSpatialRef()
crs_par = parcel_lyr.GetSpatialRef()

# define all transformations
transform_grows = osr.CoordinateTransformation(crs_par , crs_grows)
transform_dem = osr.CoordinateTransformation(crs_par , crs_dem)
transform_roads = osr.CoordinateTransformation(crs_par , crs_roads)
transform_public = osr.CoordinateTransformation(crs_par , crs_public)
transform_timber = osr.CoordinateTransformation(crs_par , crs_timber)

# copy function from assignment for to translate coordinates in array indices
def image_offsets(img, coordinates):
    gt = img.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    offsets_ul = list(map(int, gdal.ApplyGeoTransform(inv_gt, coordinates[0], coordinates[1])))
    offsets_lr = list(map(int, gdal.ApplyGeoTransform(inv_gt, coordinates[2], coordinates[3])))
    return offsets_ul + offsets_lr

# create empty dataframe with specified colums and empty lists inside
summary = pd.DataFrame(columns={'Parcel APN': [],
                                'Nr_GH-Plants': [],
                                'Nr_OD-Plants': [],
                                'Dist_to_grow_m': [],
                                'KM Priv. Road': [],
                                'KM local Road': [],
                                'Mean Elevation': [],
                                'PublicLand-YN': [],
                                'Prop_i n_THP': []
                                })

# start loop and loop over parcels
for parcel in tqdm(parcel_lyr):

    APN = parcel.GetField('APN')  # Get APN identifier

    # ------------ GH-grows / outdoor-grows ---------
    geom = parcel.GetGeometryRef().Clone()  # Get Geometry from Parcel
    geom.Transform(transform_grows)  # Transform utilizing transform definition above *
    mar_lyr.SetSpatialFilter(geom)  # Spatial selection
    greenhouse_hash = 0  # set counter to 0
    outside_hash = 0  # set counter to 0
    # loop through each cannabis count in cropped layer
    for grow in mar_lyr:
        greenhouse_hash += grow.GetField('g_plants')  # add number to previous value as long as in same parcel APN
        outside_hash += grow.GetField('o_plants')  # same here (+=)
    mar_lyr.SetSpatialFilter(None)  # reset spatial filter
    mar_lyr.ResetReading()
    # ------------------------------------------------

    # ----------- Distance of a parcel to  next grow (in m) --------
    geom = parcel.GetGeometryRef().Clone()
    mar_lyr.SetSpatialFilter(geom)
    grow_inside = mar_lyr.GetFeatureCount()
    # set a tempory variable to store the number of grows inside each buffer
    grow_check = grow_inside
    # if there's none grow in a parcel, then result should be zero
    if grow_inside == 0 :
        buffsize = 0
    else :
        # if there's grow in a parcel,
        while grow_check == grow_inside :
            # if there's no adding points in the buffer, then enlarge the buffer size.
            buffsize += 10
            buffer_geom = geom.Buffer(buffsize)
            mar_lyr.SetSpatialFilter(buffer_geom)
            grow_check = mar_lyr.GetFeatureCount()
    mar_lyr.SetSpatialFilter(None)
    # ------------------------------------------------

    # --------- km of private and local roads --------
    geom = parcel.GetGeometryRef().Clone()
    geom.Transform(transform_roads)
    local = 0                                           # initiate counter for local and private with zero
    private = 0
    roads_lyr.SetSpatialFilter(geom)                    # filter roads to parcel geometry
    if roads_lyr.GetFeatureCount() > 0:                 # if there are roads start loop
        for road in roads_lyr:
            road_type = road.GetField('FUNCTIONAL')     # get feature attributes
            geom_road = road.geometry()                 # get geometry of road feature
            #crop roads to parcel
            if road_type == 'Local Roads':              # if feature is local road then calculate length
                local = geom_road.Intersection(geom).Length() / 1000 # length only for intersection part
            if road_type == 'Private' :
                private = geom_road.Intersection(geom).Length() / 1000
    roads_lyr.SetSpatialFilter(None)
    roads_lyr.ResetReading()
    # ------------------------------------------------

    # ------------ mean elevation of parcel ---------
    # Geometry of each parcel
    geom = parcel.GetGeometryRef().Clone()
    geom.Transform(transform_dem)
    x_min , x_max , y_min , y_max = geom.GetEnvelope()

    # create temporary shape file with only the current feature
    outDataSource = outDriver.CreateDataSource('temp.shp')
    outLayer = outDataSource.CreateLayer('' , crs_dem , geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    # create feature
    feat = ogr.Feature(featureDefn)
    feat.SetGeometry(geom)
    outLayer.CreateFeature(feat)

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
    feat = None
    outDataSource = None
    outLayer = None
    # read mask
    mask = band.ReadAsArray()
    # get array indices for subsetting the dem
    img_offsets = image_offsets(dem , (x_min , y_max , x_max , y_min))
    # subset dem
    dem_sub = dem.ReadAsArray(img_offsets[0] , img_offsets[1] , mask.shape[1] , mask.shape[0])

    # apply mask and calc mean
    dem_mean = np.mean(np.where(mask == 1 , dem_sub , 0))
    # ------------------------------------------------


    # ------------ Public Land-YN ---------
    # parcel lies on public land = 1
    # parcel does not lay on public land = 0
    withinpublic = 0  # instatiate integer with 0
    geom_par = parcel.geometry()  # get geometry of individual parcel
    publiclands_lyr.SetSpatialFilter(geom_par)  # spatially filter with public lands
    if publiclands_lyr.GetFeatureCount() > 0 :  # if parcel outside public land layer -> featurecount would be 0
        withinpublic = 1  # if parcel within public lands (featurecount >0) set variable to 1
    publiclands_lyr.SetSpatialFilter(None)  # reset spatial filter for next parcel
    # ------------------------------------------------

    # ------------Proportion of parcel in THP. ---------
    geom = parcel.GetGeometryRef().Clone()
    geom.Transform(transform_timber)
    area_parcel = geom.GetArea()
    timber_lyr.SetSpatialFilter(geom)
    area_timber = 0                             # initiate area variable with zero
    if timber_lyr.GetFeatureCount() > 0:        # if there is a timber feature inside the parcel -> start loop
        #loop through all timber areas and calculate the area
        for timber in timber_lyr:
            geom_timber = timber.GetGeometryRef()
            intersection = geom_timber.Intersection(geom)
            area_timber += intersection.GetArea()
        timber_lyr.ResetReading()
    #calculate proportion
    percentage_timber = area_timber / area_parcel
    #if there are more than 1 timber features in one 1 parcel the proportion could be larger than 1
    if percentage_timber > 1:
        percentage_timber = 1
    # ------------------------------------------------------


    # ------------ write variable values to dataframe ---------
    summary = summary.append({'Parcel APN' : APN ,
                              'Nr_GH-Plants' : greenhouse_hash ,
                              'Nr_OD-Plants' : outside_hash ,
                              'Dist_to_grow_m' : buffsize ,
                              'KM Priv. Road' : private,
                              'KM local Road' : local,
                              'Mean Elevation' : dem_mean ,
                              'PublicLand-YN' : withinpublic ,
                              'Prop_i n_THP' : percentage_timber} , ignore_index=True)
    # ----------------------

# parcel_lyr.ResetReading()
summary.to_csv("summary.csv",index=False, float_format='%.2f', encoding='utf-8-sig')
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