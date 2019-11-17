# ############################################################################################################# #\
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #\
# encouraged to modify the scripts to her/his own needs.                                                        #\
# (c) Vincent Hanuschik, 22/10/2019                                            #\
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #\

import time
import os
import glob

# ####################################### SET TIME-COUNT ###################################################### #\
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")

# ####################################### FOLDER PATHS & global variables ##################################### #
landsat_dir = "/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment02_data/Part01_Landsat/"
shp_dir = "/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment02_data/Part02_GIS-Files/"
footprints = os.listdir(landsat_dir)
gis_files = os.listdir(shp_dir)  # list all files in shp_dir

# ############################################################################################################# #\
# ####################################### Exercise 1-sanity check ############################################# #\
# ############################################################################################################# #\
# assess for each footprint, how many scenes from each individual sensor are located in the folders

for folder in footprints :                                                         #loop through footprints
    path_row = landsat_dir + folder                                                # add folder to landsat_dir string
    print('Path/Row' , folder)
    print("No of Landsat 8 scenees:" , len(glob.glob1(path_row , 'LC08*')))        # check the numer of subfolders,
    print("No of Landsat 7 scenees:" , len(glob.glob1(path_row , 'LE07*')))        # starting with specified string,
    print("No of Landsat 5 scenees:" , len(glob.glob1(path_row , 'LT05*')))        # uses wildcard * in order to
    print("No of Landsat 4 scenees:" , len(glob.glob1(path_row , 'LT04*')) , '\n') # match first four charactes only


# count the number of scenes that do not have the “correct” number of files in them
corrupt_scenes = open('assignment02_corrupt_scenes.txt' , 'w')
for footprint in footprints :                                                   # iterates the footprint folders
    path_footprint = os.path.join(landsat_dir , footprint)                      # string with path of footprint
    scenes = os.listdir(path_footprint)                                         # lists scenes within each footprint
    for scene in scenes :                                                       # iterates scene within each footprint
        path_scene = os.path.join(path_footprint , scene)                       # string with path of scene
        files = os.listdir(path_scene)                                          # lists files within eachscene
        if scene.startswith('LC08') == True :                # checks if first 4 characters in match 'LC08' individually
            if len(files) < 19 :                             # checks if number of files is less than 19
                corrupt_scenes.write(str(os.path.join(footprint , scene))+'\n')
        elif scene.startswith('LE07') == True :
            if len(files) < 19 :
                corrupt_scenes.write(str(os.path.join(footprint , scene))+'\n')
        elif scene.startswith('LT05') == True:
            if len(files) < 21 :
                corrupt_scenes.write(str(os.path.join(footprint , scene))+'\n')
        elif scene.startswith('LT04') == True:
            if len(files) < 21 :
                corrupt_scenes.write(str(os.path.join(footprint , scene))+'\n')
corrupt_scenes.close()                                                  # close file

# ############################################################################################################# #\
# ############################################## FUNCTION ##################################################### #\
# ############################################################################################################# #\

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


# ############################################################################################################# #\
# ####################################### Exercise 2-sanity check ############################################# #\
# ############################################################################################################# #\
# number of all files in shp_dir with specific extension (.shp & .tif) without file path
len(ListFiles(shp_dir , '.shp' , 0))
len(ListFiles(shp_dir , '.tif' , 0))

# ################################### CHECK FOR INCOMPLETE SHAPEFILES ########################################## #\

### WORKFLOW: Get filenames without extensions -> add mandatory extensions -> add directory ###
### -> Check if os.path.isfile == TRUE --> if not true --> write missing filename to file   ###

ext_shp = ('.shp', '.shx', '.dbf', '.prj')                  # tuple with extensions mandatory for working shapefiles
shp_list = ListFiles(shp_dir , '.shp' , 0)                  # Lists all files in shp_dir with .shp extension
shp_names = [os.path.splitext(shp)[0] for shp in shp_list]  # splits the shapefile from its extension -> name only

shp_exists = [shp + ext for ext in ext_shp for shp in shp_names]  # creates filenames with mandatory extension
shp_exists_path = [shp_dir + shp  for shp in shp_exists ]      # creates filepaths for all filenames per shapefile-name

missing_vector_files = open('assignment02_missing_vector_files.txt', 'w')   # creates new .txt file
for shp in shp_exists_path:                                                 # loops through all hypothetical file paths
    if os.path.isfile(shp) == False:                                        # if the path is not a file path
        missing_vector_files.write(str(os.path.basename(shp)+ '\n'))        # write the basename (e.g. filename) of the
        # print(str(os.path.basename(shp)+ '\n'))                           # missing file to the .txt
missing_vector_files.close()                                                # close file

# ####################################### END TIME-COUNT AND PRINT TIME STATS################################## #\
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")