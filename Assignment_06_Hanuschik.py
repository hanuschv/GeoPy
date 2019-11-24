# ==================================================================================================== #
#   Assignment_06                                                                                      #
#   (c) Vincent Hanuschik, 19/11/2019                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import gdal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import colors
# ======================================== SET TIME COUNT ============================================ #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
data = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment06_data/'
forest_msk = data +'gpy_poland_forestmask.tif'
vertex = data +'gpy_poland_landtrendr_tcw_vertex_8618.tif'
# =========================================== FUNCTIONS ============================================== #
def plotDisturbanceYearMap(img):
    values = np.unique(img.ravel())
    values = values[values > 0]

    cmap = plt.get_cmap('jet', values.size)
    cmap.set_under('grey')
    norm = colors.BoundaryNorm(list(values), cmap.N)

    plt.figure(figsize=(8, 4))
    im = plt.imshow(img, interpolation='none', cmap=cmap, norm=norm)

    dist_colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=dist_colors[i], label=values[i]) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
               borderaxespad=0., ncol=2, frameon=False)

    plt.grid(True)
    plt.show()
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
# ========================================== PROCESSING ============================================== #

# open vertex file and forest mask as arrays (Forest mask: 1 = forest, 0 = non-forest)
vertex_arr = gdal.Open(vertex).ReadAsArray()
fmask = gdal.Open(forest_msk).ReadAsArray()

# extract array with years and extract array with fitted values
vertex_years = vertex_arr[1:7] # first year is not needed
vertex_fitted = vertex_arr[14:21]

# calculate change between timesteps along axis 0 (pixel over time) and take the max value
vdiff = np.diff(vertex_fitted, axis = 0)
vdiffmax = np.nanmax(vdiff, axis = 0)

# take index of max change magnitude, put in array the corresponding year (np.choose)
# multiply year with 1 (forest) or 0 (non-forest) if change magintude > 500 (True | False)
# results in array with years or 0
vdiffargmax = np.nanargmax(vdiff, axis = 0)
vidxyear = np.choose(vdiffargmax, vertex_years)
yearvdiffmax_masked = fmask*vidxyear*(vdiffmax>500)

# plot the disturbance
plotDisturbanceYearMap(yearvdiffmax_masked)

# write array to disk using the vertex as gdal reference
array2geotiff(yearvdiffmax_masked, 'yearvdiffmax_masked.tif',ingdal=gdal.Open(vertex), dtype= gdal.GDT_Int16)
# =============================== END TIME-COUNT AND PRINT TIME STATS ============================== #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")