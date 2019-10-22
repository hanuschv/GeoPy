# ############################################################################################################# #\
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #\
# encouraged to modify the scripts to her/his own needs.                                                        #\
# (c) Vincent Hanuschik, 22/10/2019                                            #\

# ####################################### LOAD REQUIRED LIBRARIES ############################################# #\
import time
import os
import math as m
import shutil
import re
import glob

# ####################################### SET TIME-COUNT ###################################################### #\
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")


# ####################################### FOLDER PATHS & global variables ##################################### #
landsat_dir = "/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment02_data/Part01_Landsat/"
#shp_dir = "/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment02_data/Part02_GIS-Files"

# ####################################### FUNCTIONS ########################################################### #\
footprints = os.listdir(landsat_dir)

for folder in footprints:
    path_row = landsat_dir + folder
    print('Path/Row', folder)
    nLC08 = len(glob.glob1(path_row,'LC08*'))
    nLE07 = len(glob.glob1(path_row, 'LE07*'))
    nLT05 = len(glob.glob1(path_row, 'LT05*'))
    nLT04 = len(glob.glob1(path_row, 'LT04*'))

    print("No of Landsat 8 scenees:", nLC08)
    print("No of Landsat 7 scenees:", nLE07)
    print("No of Landsat 5 scenees:", nLT05)
    print("No of Landsat 4 scenees:", nLT04, "\n")

os.listdir('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/')

#os.makedrs(folder + "/test/")

next(os.walk(usr/lib))

# ####################################### PROCESSING ########################################################## #\
\
\
\
\
# ####################################### END TIME-COUNT AND PRINT TIME STATS################################## #\




