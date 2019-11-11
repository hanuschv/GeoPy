# ==================================================================================================== #
#   Assignment_04                                                                                      #
#   (c) Vincent Hanuschik, 10/11/2019                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import gdal
import numpy as np
# ======================================== SET TIME COUNT ============================================ #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
path = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment04_Files/'

# sorted() sorts alphabetically in order to maintain clarity. file_list[0]=2000; file_list[1]=2005 etc.
file_list = [path + file for file in sorted(os.listdir(path))]
# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise I ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #

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

def overlapExtent(rasterPathlist):
    '''
    Finds the common extent/ overlap and returns geo coordinates from the extent.
    Returns list with corner coordinates.
    Uses GetGeoTransform to extract corner values of each raster.
    Common extent is then calculated by maximum ul_x value,
    minimum ul_y value, minimum lr_x value and maximum lr_y value.
    Use list comprehensions to calculate respective coordinates for all rasters and
    use the index to extract correct position coordinates[upperleft x, upperleft y, lowerright x and lowerright y].
    :param rasterPathlist:
    :return:
    '''
    ul_x_list = [CornerCoordinates(path)[0] for path in rasterPathlist]
    ul_y_list = [CornerCoordinates(path)[1] for path in rasterPathlist]
    lr_x_list = [CornerCoordinates(path)[2] for path in rasterPathlist]
    lr_y_list = [CornerCoordinates(path)[3] for path in rasterPathlist]
    overlap_extent = []
    overlap_extent.append(max(ul_x_list))
    overlap_extent.append(min(ul_y_list))
    overlap_extent.append(min(lr_x_list))
    overlap_extent.append(max(lr_y_list))
    return overlap_extent

geo_overlap = overlapExtent(file_list)
# print(geo_overlap)
# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise I & II ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
imglist = [gdal.Open(raster) for raster in file_list]

def subset(raster, subset, coordinates=True, multiband=False):
    """
    Subsets a given raster to a set of coordinates [subset] into an array.
    When subset is given as list of geographic coordinates, they will be transformed to array indices.
    When subset is given as array indices, image will be subsetted. Indices must be in format [ulx, uly,lrx,lry]
    Bool-Arguments don't have to be specified, when called. Defaults to function definition.
    :param raster:
    :param subset:
    :param coordinates:
    :param multiband:
    :return array:
    """
    if coordinates:
        gt = raster.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        app_gt_offset_upperleft = gdal.ApplyGeoTransform(inv_gt , geo_overlap[0] , geo_overlap[1])
        app_gt_offset_lowerright = gdal.ApplyGeoTransform(inv_gt , geo_overlap[2] , geo_overlap[3])
        off_ulx , off_uly = map(int , app_gt_offset_upperleft)
        off_lrx , off_lry = map(int , app_gt_offset_lowerright)
        rows = off_lry - off_uly
        columns = off_lrx - off_ulx
        # idx = [off_ulx, off_uly, columns, rows ]
        array = raster.ReadAsArray(off_ulx , off_uly , columns , rows)
        return array
    else:
        idx = subset

    if gdal:
        array = raster.ReadAsArray(idx[0], idx[1],
                                   idx[2] - idx[0],
                                   idx[3] - idx[1])
    else:
        if multiband:
            array = raster[:, idx[0]:idx[2], idx[1]:idx[3]]
        else:
            array = raster[idx[0]:idx[2], idx[1]:idx[3]]
    return array

# stack_sub = [subset(raster, geo_overlap) for raster in stacklist]

#applies subset function to all images in imglist [see above]. Returns list of respective arrays.
sublist = [subset(img, geo_overlap, coordinates=True) for img in imglist]


#small summary function to calculate statistics for each array.
def arr_summary(array, decimals = 2):
    '''
    calculates maximum, mean, minimum and standard deviation values for given array.
    Decimals specified are used only for mean and std.
    :param array:
    :param decimals:
    :return:
    '''
    max = ['maximum value' ,np.max(array)]
    mean = ['mean value', round(np.mean(array), decimals)]
    min = ['minimum value', np.min(array)]
    std = ['standard deviation value' , round(np.std(array), decimals)]
    return max, mean, min, std

summary = [arr_summary(arr, 2) for arr in sublist]
# [print(sum) for sum in summary]

# =============================== END TIME-COUNT AND PRINT TIME STATS ============================== #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")