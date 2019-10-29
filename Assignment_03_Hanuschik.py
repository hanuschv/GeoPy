# ############################################################################################################# #\
#                                                                                                               #\
#                                                                                                               #\
# (c) Vincent Hanuschik, DD/MM/YYYY                                                                             #\
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #\
import time
import os
import pandas as pd

# ####################################### SET TIME-COUNT ###################################################### #\
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")

# ####################################### FOLDER PATHS & global variables ##################################### #
data_dir =  '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment03_data/'
landsat_df = pd.read_csv(data_dir+'LANDSAT_8_C1_313804.csv', index_col=0)
lucas_df = pd.read_csv('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment03_data/DE_2015_20180724.csv', index_col=0)
grid_df = pd.read_csv(data_dir+'GRID_CSVEXP_20171113.csv', index_col=0)


# ############################################################################################################# #\
# ####################################### Exercise XXXXXXXXXXXXXX ############################################# #\
# ############################################################################################################# #\

# OBS_TYPE = 1
# OBS_DIRECT = 1
# OBS_RADIUS <= 2
# AREA_SIZE >= 2
# FEATURE_WIDTH > 1
# LC1_PCT >= 5
lucas_df_red = lucas_df[(lucas_df.OBS_TYPE == 1) & (lucas_df.OBS_DIR == 1) & (lucas_df.OBS_RADIUS <= 2) &
                        (lucas_df.AREA_SIZE >= 2) & (lucas_df.FEATURE_WIDTH > 1) & (lucas_df.LC1_PCT >= 5)]
nlucas = len(lucas_df_red)

nlc1 = len(lucas_df_red.LC1.unique())
meanobsdist = lucas_df_red.OBS_DIST.mean()
meanobsdista21 = lucas_df_red.OBS_DIST.mean(lucas_df_red.LC1 == A21)

# inplace=True


# ############################################################################################################# #\
# ############################################## FUNCTION ##################################################### #\
# ############################################################################################################# #\


# ####################################### END TIME-COUNT AND PRINT TIME STATS################################## #\
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")