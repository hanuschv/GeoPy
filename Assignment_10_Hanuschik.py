# ==================================================================================================== #
#   Assignment_10                                                                                      #
#   (c) Vincent Hanuschik, 09/01/2020                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import gdal, ogr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osr
# ======================================== SET TIME COUNT ============================================ #
time_start = time.localtime()
time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_start)
print("--------------------------------------------------------")
print("Start: " + time_start_str)
print("--------------------------------------------------------")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
os.chdir('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment10_data/data')
landcover = gdal.Open('landcover_lucas2015_15000_10000.tif')
landsat = gdal.Open('landsat_median1416_15000_10000.tif')
lucas = ogr.Open('EU28_2015_20161028_lucas2015j.gpkg')
# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
def random_sample(a, a_value=1, n_samples=1000, min_distance=1):
    sample_mask = np.zeros_like(a, dtype=bool)
    sample_mask[::min_distance, ::min_distance] = True

    if a_value is not None:
        a_bool = np.where(a == a_value, True, False)
        sample_mask = np.where(a_bool * sample_mask)

    sample = np.random.choice(np.arange(0, len(sample_mask[0]), 1), size=n_samples, replace=False)
    choices = [tuple(sample_mask[0][sample]), tuple(sample_mask[1][sample])]

    mask = np.zeros_like(a, dtype=bool)
    mask[choices] = True

    return mask

def create_poly_extent(img):
    gt = img.GetGeoTransform()

    ulx = gt[0]
    uly = gt[3]
    lrx = gt[0] + gt[1] * img.RasterXSize
    lry = gt[3] + gt[5] * img.RasterXSize
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

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise XX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
np.random.seed(42)

landcover_arr = landcover.ReadAsArray()
landsat_arr = landsat.ReadAsArray()

# create summary data-frame
summary = pd.DataFrame(columns={'classID':[],
                                'B':[],
                                'G':[],
                                'R': [],
                                'NIR': [],
                                'SWIR1': [],
                                'SWIR2': [],
                                }, dtype=float)

# get unique land cover classes
landcover_classes = np.unique(landcover_arr)

# translate minimum distance of 500m into Landsat pixels to use in random_sample function
min_distance = 17

# iterate over each class and apply random sampling on raster
for uclass in landcover_classes:
    # create random sample mask
    # try with minimum distance, but lower if population size is not sufficient (-> except)
    while True:

        try:
            mask = random_sample(landcover_arr , uclass , n_samples=1000 , min_distance=min_distance)
            extracted = landsat_arr[:, mask].T
            class_array = np.full(extracted.shape[0] , uclass).reshape(-1 , 1)
            extracted = np.concatenate([class_array, extracted], axis=1)

            # add extracted values to dataframe
            summary = pd.concat([summary, pd.DataFrame(extracted, columns=summary.columns)], axis=0)

        except ValueError:
            min_distance = min_distance-1
            print('Class ' + str(uclass) + ': min_distance has been lowered by -1 to ' + str(min_distance))
            if min_distance < 1:
                print('Class: ' + str(uclass))
                print('Chosen sample size larger than population size!')
                break
            continue

        break

# calculate spectral means for each class
class_means = summary.groupby("classID").mean()
class_means.to_csv('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/LUCAS_sampled_class_mean.csv')


# plot result
plt.figure()
class_means.T.plot()
plt.show()


# ================================= Excercise II ==================================================



# Get extent of Landsat data and transform to LUCAS geotransform
extent = create_poly_extent(landsat)
landsat_crs = osr.SpatialReference(wkt = landsat.GetProjection())
gt = landsat.GetGeoTransform()
transformer = osr.CoordinateTransformation(landsat_crs, lucas.GetLayer().GetSpatialRef())
extent.Transform(transformer)

# "SetSpatialFilter" on LUCAS dataset using transformed extent
lyr_lucas = lucas.GetLayer()
lyr_lucas.SetSpatialFilter(extent)

backtransformer = osr.CoordinateTransformation(lucas.GetLayer().GetSpatialRef(), landsat_crs)

# create summary dataframe
summary = pd.DataFrame(columns={'class':[],
                                'x':[],
                                'y':[],
                                'band2':[],
                                'band3':[],
                                'band4': [],
                                'band5': [],
                                'band6': [],
                                'band7': [],
                                })


for feat in lyr_lucas:

    geom = feat.GetGeometryRef().Clone()
    x, y = geom.GetX(), geom.GetY()
    geom.Transform(backtransformer)
    mx, my = geom.GetX(), geom.GetY()
    offx = int((mx - gt[0]) / gt[1])  # x pixel
    offy = int((my - gt[3]) / gt[5])  # y pixel

    lc_value = landcover.ReadAsArray(offx , offy , 1 , 1).flatten()
    landsat_values = landsat.ReadAsArray(offx, offy, 1, 1).flatten()

    summary = summary.append({'Class': int(lc_value),
                              'X': x,
                              'Y': y,
                              'Band2': landsat_values[0],
                              'Band3': landsat_values[1],
                              'Band4': landsat_values[2],
                              'Band5': landsat_values[3],
                              'Band6': landsat_values[4],
                              'Band7': landsat_values[5]}, ignore_index=True)

summary.to_csv('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/LUCAS_Landsat_extracted.csv')






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