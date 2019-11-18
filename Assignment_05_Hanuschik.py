# ==================================================================================================== #
#   Assignment_05                                                                                      #
#   (c) Vincent Hanuschik, 12/11/2019                                                                  #
#                                                                                                      #
#                Informer:                                                                             #
#                I definitely underestimated the time it takes to write the last                       #
#                part of the composite function in order to write the array to                         #
#                geoTIFF. I read into the topic but then chose not to try to                           #
#                write it, because of time constraints. I rather wanted to have                        #
#                a working code until that point so i could work on the geoTiff                        #
#                part later. Apologies.                                                                #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import numpy as np
import gdal
# ======================================== SET TIME COUNT ============================================ #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ================================ DATA PATHS & DIRECTORIES ========================================== #
dir = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment05_landsat8_2015/'
footprints = sorted(os.listdir(dir))
folder_list = [dir+footprint + '/' for footprint in footprints if not(footprint.startswith('.'))]
# ======================================== FUNCTIONS ================================================ #

# taken from previous assignments
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

def qa_mask(qa_path):
    '''
    Creates Mask based on the criteria applied (good observations = clear land (0), clear water(1),  snow(3)
    Where conditions is True -> make array value 1, otherwise zero.
    Returns array with 1's and 0's.
    :param qa_path:
    :return:
    '''
    bqa = gdal.Open(qa_path).ReadAsArray()
    mask = np.where(((bqa == 0) | (bqa == 1) | (bqa == 3)), 1, 0)
    return mask

def findfileinfolder(filepath, ending):
    '''
    Takes a filepath and looks for a certain ending in the same folder.
    Returns the file.
    :param filepath:
    :param ending:
    :return:
    '''
    folder = os.path.dirname(filepath)+ '/'
    files = os.listdir(folder)
    for f in files:
        if f.endswith(ending):
            file = folder + f
    return file

def maskImg(sr_path):
    '''
    Apply mask to img. Automatically finds the corresponding qa_file.
    Returns array with masked image, where masked values are 0.
    :param sr_path:
    :return:
    '''
    img = gdal.Open(sr_path).ReadAsArray()
    qa_file = findfileinfolder(sr_path, 'qa_clip.tif')
    mask = qa_mask(qa_file)
    masked = img*mask                                   # since mask is 1 or 0 img*mask returns img value or 0
    return masked
    # apply mask from corresponding filepath

def meanimg(sr_path_list):
    '''
    Stacks all files in file- list and applies nanmean along axis 0 (first dimension of array).
    Nanmean ignores nan's if there are any.
    Returns the array with mean values.
    :param sr_path_list:
    :return:
    '''
    stack = np.stack([maskImg(sr) for sr in sr_path_list])
    meanimg = np.nanmean(stack, axis=0)
    return meanimg

def calcNdVI(nir, red):
    '''
    Calculates NDVI without storing values. Simply returns the value
    :param nir:
    :param red:
    :return:
    '''
    return (nir - red) / (nir + red)

def NDVI_comp(sr_path_list):
    '''
    Creates a composite with original band values where the ndvi is maximum using argmax().
    Only includes good observations for NDVI calculation, using maskImg.
    Returns composite.
    :param sr_path_list:
    :return:
    '''
    scenes = [maskImg(scene) for scene in (sr_path_list)]
    scenes_stack = np.stack(scenes , axis=0)
    ndvi_stack = np.stack([calcNdVI(scene[3], [2]) for scene in scenes])   #calculates NDVI for each scene and stacks it
    ndvi_max = np.nanargmax(ndvi_stack, axis=0)
    ndvi_comp = np.choose(ndvi_max, scenes_stack)  # take values from scenes_stack (original values) from ndvi_max index
    ndvi_comp = ndvi_comp.astype(np.int16)         # convert floats to int
    return ndvi_comp

# ======================================== PROCESSING ================================================== #
sr_files = [ListFiles(sub, 'sr_clip.tif', 1) for sub in folder_list]
#ListFiles() creates List for each item. This list comprehension "flattens" it into single list.
sr_files = [val for sublist in sr_files for val in sublist]

mean_stack = meanimg(sr_files)
# print(mean_stack.shape)
ndvi_comp = NDVI_comp(sr_files)
# print(ndvi_comp.shape)

# ============================================ FINISH ================================================== #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S" , time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")