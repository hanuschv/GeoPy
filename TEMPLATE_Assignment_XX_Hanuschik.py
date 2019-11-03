# ==================================================================================================== #
#   Assignment_03                                                                                      #
#   (c) Vincent Hanuschik, 03/11/2019                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import pandas as pd
# ======================================== SET TIME COUNT ============================================ #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ==================================================================================================== #
data_dir =  '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment03_data/'
landsat_df = pd.read_csv(data_dir+'LANDSAT_8_C1_313804.csv', index_col=0)
lucas_df = pd.read_csv(data_dir+'DE_2015_20180724.csv', index_col=0)
grid_df = pd.read_csv(data_dir+'GRID_CSVEXP_20171113.csv', index_col=0)
# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise I ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #

# filters lucas_df according to the requirements. & logical AND; | logical OR
df = lucas_df[(lucas_df.OBS_TYPE == 1) & (lucas_df.OBS_DIR == 1) &
              (lucas_df.OBS_RADIUS <= 2) & (lucas_df.AREA_SIZE >= 2) &
              (lucas_df.FEATURE_WIDTH > 1) & (lucas_df.LC1_PCT >= 5)]
nlucas = len(df)                                         #number of rows in df left = samples left
nlc1 = df.LC1.nunique()                                  # number of unique values/classes in specified column
meanobsdist = df.OBS_DIST.mean()                         #mean of specified column
meanobsdistA21 = df.OBS_DIST[df.LC1 == 'A21'].mean()     # mean of specified column, where condition is applied
lc_n = df.LC1.value_counts()                             #counts number of appearances of unique values in column
                                                         # .mode() only returns class with most counts,
                                                         # not number of values counted
lc_n.iloc[0]                                             # spits out number of values per class in first row

# merges specified colums of grid_df to df into wgs_df based on internal index POINT_ID
wgs_df = df.merge(grid_df[['X_WGS84', 'Y_WGS84']], on= 'POINT_ID')

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise II ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #

#renames colums in landsat df: str.lower() converts upper case to lower case,
# str.replace('str in column' converted to 'new str'); can be concatenated in one line
landsat_df.columns = landsat_df.columns.str.lower()\
    .str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')

#filter landsat_df based on criteria
l8_df = landsat_df[(landsat_df.land_cloud_cover < 70)&
                   (landsat_df.day_night_indicator == 'DAY') &
                   (landsat_df.data_type_level_1 == 'OLI_TIRS_L1TP')]
#calculate averages of geometric accuracies x/y and land cloud cover
mean_geom_acc_x = l8_df.geometric_rmse_model_x.mean()
mean_geom_acc_y = l8_df.geometric_rmse_model_y.mean()
mean_lCC = l8_df.land_cloud_cover.mean()

#get number of unique path/row combinations by concatenating values as strings (astype(str)
# into new column sperated by '_'; then return number of unique strings in that column (.nunique)
l8_df['path_row'] = l8_df['wrs_path'].astype(str) + '_' + l8_df['wrs_row'].astype(str)
l8_df.path_row.nunique()

# =============================== END TIME-COUNT AND PRINT TIME STATS ============================== #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")