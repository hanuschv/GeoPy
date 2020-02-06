# ==================================================================================================== #
#   MAP                                                                                      #
#   (c) Vincent Hanuschik, 10/12/2019                                                                  #
#
#
#   # tested and running on a Mac running Python
#
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import gdal
import ogr, osr
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


# =======================================  Part 1 ==================================================== #
pointfiles = ListFiles(pointdir, '.shp', 1)
pointshp = ogr.Open(pointfiles[0])

rasterfiles = ListFiles(rasterdir, '.tif', 1)
rastertiles = [gdal.Open(tile) for tile in rasterfiles]
rasterarr = [gdal.Open(tile).ReadAsArray() for tile in rasterfiles]

pointlyr = pointshp.GetLayer()

# get Projections of Layers to check coordinate systems. In this case both are projected using EPSG 4326/WGS84
pointcrs = pointlyr.GetSpatialRef()
rastercrs = rastertiles[0].GetProjection()

'''
tile0 = gdal.Open(rasterfiles[0])
extent = create_poly_extent(tile0)
landsat_crs = osr.SpatialReference(wkt = tile0.GetProjection())
gt = tile0.GetGeoTransform()
transformer = osr.CoordinateTransformation(landsat_crs, pointshp.GetLayer().GetSpatialRef())
extent.Transform(transformer)
# "SetSpatialFilter" on LUCAS dataset using transformed extent
lyr_pt = pointshp.GetLayer()
lyr_pt.SetSpatialFilter(extent)
'''



all_values= list()
for raster in rasterfiles:
    tile = gdal.Open(raster)
    tile_values = list() # create list for each tile, where values of point response variable is stored
    # open tile
    tile_arr = tile.ReadAsArray()

    # Get Extent from raster tile and create Polygon extent
    poly_extent = create_poly_extent(tile)
    # Select points within tile using SpatialFilter()
    pointlyr_tile = pointlyr.SetSpatialFilter(poly_extent.GetGeometry()) # not working at the moment

    for point in pointlyr_tile:
        print(point.GetField('CID'))
        geom = point.GetGeometryRef()
        feat_id = point.GetField('id_points')
        mx, my = geom.GetX(), geom.GetY()

        # translate point coordinates to image/array coordinates
        # px = int((mx - gt[0]) / gt[1])
        # py = int((my - gt[3]) / gt[5])

        # intval = px.ReadAsArray(px, py, 1, 1)
        tile_values.append([feat_id, intval[0]])

    # append list with extracted raster values to global list of values (all tiles)
    # close tile


# pyramid layer level all (2-32) use gdaladdo
#
# Create directory
dirName = 'tempDir'

try :
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName , " Created ")
except FileExistsError :
    print("Directory " , dirName , " already exists")




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