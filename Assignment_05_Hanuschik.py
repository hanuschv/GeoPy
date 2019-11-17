# ==================================================================================================== #
#   Assignment_05                                                                                      #
#   (c) Vincent Hanuschik, 12/11/2019                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import numpy as np
import gdal
import matplotlib as plt

# ======================================== SET TIME COUNT ============================================ #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
# path = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment05_landsat8_2015/LC81930232015164LGN00/LC81930232015164LGN00_qa_clip.tif'
# img = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment05_landsat8_2015/LC81930232015164LGN00/LC81930232015164LGN00_sr_clip.tif'
# folder = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment05_landsat8_2015/LC81930232015164LGN00/'

dir = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment05_landsat8_2015/'
footprints = os.listdir(dir)
folder_list = [dir+footprint + '/' for footprint in footprints if not(footprint.startswith('.'))]

files = [os.path.dirname(folder) + os.listdir(folder) for folder in folder_list if not(folder.startswith('.'))]
sr_files = [ for sr in files if (sr.endswith('qa_clip.tif'))]
"Array slicing creates a View on the original array - NOT a copy"

# logicalmask = cloudscore > 42
# ndvi[logicalmask]               # takes value, where array bool == True
# ndvi[~logicalmask]              #inverts mask ->takes value, where array bool == False

# plt.imshow()
# plt.show()
# qa_clip.tif: one single band/layer raster with the following pixel codes:
# 0 => clear land pixel, 1 => clear water pixel, 2 => cloud shadow, 3 => snow, 4 => cloud, 255 => fill value.

def qa_mask(qa_path):
    bqa = gdal.Open(qa_path).ReadAsArray()
    mask = np.where(((bqa == 0) | (bqa == 1) | (bqa == 3)), 1, 0)
    return mask

def maskImg(sr_path):
    img = gdal.Open(sr_path).ReadAsArray()
    folder = os.path.dirname(sr_path)+ '/'
    imgpath = os.listdir(folder)
    for f in imgpath:
        if f.endswith('qa_clip.tif'):
            qa = folder + f
    mask = qa_mask(qa)
    masked = np.where((mask == 1), img, np.nan)
    return masked
    # apply mask from corresponding filepath

def meanimg(sr_path_multiple):
    for sr in sr_path_multiple:
        masked = maskImg(sr)
    return list

    # stack 17 images with 6 bands
    # np.mean(axis= 0) along time (17footprints), account for NaN

def NDVI(filepath):



# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise X ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #


# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise XX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #



# =============================== END TIME-COUNT AND PRINT TIME STATS ============================== #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")