# ############################################################################################################# #\
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #\
# encouraged to modify the scripts to her/his own needs.                                                        #\
# (c) Vincent Hanuschik, 22/10/2019                                            #\

# ####################################### LOAD REQUIRED LIBRARIES ############################################# #\
import time
import os
import shutil
import glob
import fnmatch

# ####################################### SET TIME-COUNT ###################################################### #\
starttime = time.strftime ("%a, %d %b %Y %H:%M:%S" , time.localtime ())
print ("--------------------------------------------------------")
print ("Starting process, time: " + starttime)
print ("")

# ####################################### FOLDER PATHS & global variables ##################################### #
landsat_dir = "/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment02_data/Part01_Landsat/"
shp_dir = "/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment02_data/Part02_GIS-Files"
footprints = os.listdir (landsat_dir)
# if not footprint.startswith('.'):
#     footprints = os.listdir(landsat_dir)

# ############################################################################################################# #\
# ####################################### Exercise 1-sanity check ############################################# #\
# ############################################################################################################# #\

# ###################### 1.) ################################## #\
# assess for each footprint, how many scenes from each individual sensor are located in the folders

for folder in footprints :
    path_row = landsat_dir + folder
    print ('Path/Row' , folder)
    print ("No of Landsat 8 scenees:" , len (glob.glob1 (path_row , 'LC08*')))
    print ("No of Landsat 7 scenees:" , len (glob.glob1 (path_row , 'LE07*')))
    print ("No of Landsat 5 scenees:" , len (glob.glob1 (path_row , 'LT05*')))
    print ("No of Landsat 4 scenees:" , len (glob.glob1 (path_row , 'LT04*')) , '\n')

# ###################### 2.) a & b   ################################## #\
# count the number of scenes that do not have the “correct” number of files in them

corrupt_scenes = []
for footprint in footprints :  # iterates the 9 footprint folders
    path_footprint = os.path.join (landsat_dir , footprint)  # string with path of footprint
    scenes = os.listdir (path_footprint)  # lists scenes within each footprint
    for scene in scenes :  # iterates scene within each footprint
        path_scene = os.path.join (path_footprint , scene)  # string with path of scene
        files = os.listdir (path_scene)  # lists files within eachscene
        if scene[:4] == 'LC08' :  # checks if first 4 characters in match 'LC08' individually
            if len (files) < 19 :  # checks if number of files is less than 19
                corrupt_scenes.append (path_scene)  # appends the path with corrupt scene to corrupt_scenes list
        elif scene[:4] == 'LE07' :
            if len (files) < 19 :
                corrupt_scenes.append (path_scene)
        elif scene[:4] == 'LT05' :
            if len (files) < 21 :
                corrupt_scenes.append (path_scene)
        elif scene[:4] == 'LT04' :
            if len (files) < 21 :
                corrupt_scenes.append (path_scene)

with open ('assignment02_corrupt_scenes.txt' , 'w') as f :  # creates txt file and imports it as opject for use; 'w' overwrites, 'a' appends
    for scene in corrupt_scenes :  # loops through scene in list of corrupt_scenes
        f.write (
            '%s\n' % scene)  # '%s\n' specifies that new lines are strings that counts for % scene in corrupt scenes


# ############################################################################################################# #\
# ############################################## FUNCTION ##################################################### #\
# ############################################################################################################# #\

def ListFiles(filepath , filetype , expression) :
    '''
    lists all files in a given folder path of a given file extentsion

    :param filepath: string of folder path
    :param filetype: filetype
    :param expression: 1 = gives out the file as file-path ; 0 = gives out the file as file-name only
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
gis_files = os.listdir(shp_dir)             #list all files in shp_dir


len(ListFiles(shp_dir, '.shp' , 0))         #list all files in shp_dir with .shp extension without file path
len(ListFiles(shp_dir, '.tif' , 0))         #list all files in shp_dir with .tif extension without file path


def ListFileWithName (filedirectory, filename):
    elist = []
    for file in filedirectory:
        if file.startswith(filename):
            elist.append(file)
        return elist



shp_list = ListFiles(shp_dir , '.shp' , 0)
shp_names = []
for shp in shp_list:
    shp_names.append(os.path.splitext(shp)[0])

ListFileWithName(gis_files, '05_PublicLands')

for name in shp_names:
    if ListFileWithName(gis_files, name) != 'None':
        print(ListFileWithName(gis_files, name))




# ####################################### END TIME-COUNT AND PRINT TIME STATS################################## #\
print("")
endtime = time.strftime ("%a, %d %b %Y %H:%M:%S" , time.localtime ())
print ("--------------------------------------------------------")
print ("start: " + starttime)
print ("end: " + endtime)
print ("")
