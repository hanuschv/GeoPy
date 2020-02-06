# ==================================================================================================== #
#   Assignment_12                                                                                      #
#   (c) Vincent Hanuschik, 10/12/2019                                                                  #
#
#
#    Everything works fine, except the problem with the standard scaler per tile.
#   I couldn't figure out a concept in time to calculate a standard scaler on the
#   whole landsat image and then applying that on each tile.
#
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import gdal
import numpy as np
import pandas as pd
import joblib
import multiprocessing
from matplotlib import pyplot as plt
# ======================================== SET TIME COUNT ============================================ #
time_start = time.localtime()
time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_start)
print("--------------------------------------------------------")
print("Start: " + time_start_str)
print("--------------------------------------------------------")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
dir ='/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment12_data'
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
files = ListFiles(dir,'.tif', 1)
landsat = gdal.Open('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Week11_Support-Vector-Machines/data/landsat8_metrics1416.tif').ReadAsArray()
lucas = pd.read_csv('/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Week11_Support-Vector-Machines/data/landsat8_metrics1416_samples.csv')
# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #

# PARALLEL: create worker and helper functions(for the process to be applied) to submit as jobs to CPU
# function for opening rasters into arrays, apply svm model with predict()
# joblib.parallel


# stack = np.stack([gdal.Open(dir+img).ReadAsArray() for img in files])  # stack all landsat as arrays

# define predictors and response
y = lucas['class']
X = lucas.drop(['class'], axis=1)
X = lucas.iloc[:, 1:]

# z_transform the predictor
from sklearn.preprocessing import StandardScaler
X_z = StandardScaler().fit_transform(X)

# split predictors and response into separate train and test sets
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_z, y, test_size=0.5, random_state= 42)

# define the model
from sklearn.svm import SVC # "Support vector classifier"
model = SVC()
model.get_params()

# perform gridsearch to optimize hyperparameters and get best estimator from fit()
from sklearn.model_selection import GridSearchCV    # GridSearch-CrossValidation
param_grid = {'C': [1, 5, 10, 50, 100, 1000],'gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid, cv=5, iid=False)
grid.fit(Xtrain, ytrain)
grid.best_estimator_.get_params()
svc_model = grid.best_estimator_

ydim1 = landsat.shape[1]
xdim1 = landsat.shape[2]
landsat1 = landsat.transpose(1, 2, 0).reshape((ydim1*xdim1, landsat.shape[0]))
scaler = StandardScaler()
landsat_z = scaler.fit_transform(landsat1)

arg_list = [(file, svc_model) for file in files]

def parallel_predict(list):
    model = list[1]
    # scaler = list[2]
    raster = gdal.Open(list[0])
    img = raster.ReadAsArray()

    ydim = img.shape[1]
    xdim = img.shape[2]
    landsat = img.transpose(1 , 2 , 0).reshape((ydim * xdim , img.shape[0]))
    landsat_z = scaler.fit_transform(landsat)

    classification = model.predict(landsat_z)
    rs = classification.reshape((ydim , xdim))
    outPath = list[0]
    outPath = outPath.replace(".tif" , "_classification.tif")
    array2geotiff(rs,outPath, ingdal=raster)
    return rs

# def parallel_predict_2(path):
#     img = gdal.Open(path).ReadAsArray()
#
#     ydim = img.shape[1]
#     xdim = img.shape[2]
#     landsat = img.transpose(1 , 2 , 0).reshape((ydim * xdim , img.shape[0]))
#
#     scaler = StandardScaler()
#     landsat_z = scaler.fit_transform(landsat)
#
#     classification = svc_model.predict(landsat_z)
#     rs = classification.reshape((ydim , xdim))
#     return rs

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

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise XX ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #

from joblib import Parallel, delayed
output = Parallel(n_jobs=3)(delayed(parallel_predict)(list) for list in arg_list)

plt.imshow(output[2])
plt.show()

test = [plot_c(img) for img in output]
def plot_c(classification) :
    from matplotlib import colors , patches
    lcColors = {1 : [1.0 , 0.0 , 0.0 , 1.0] ,  # Artificial
                2 : [1.0 , 1.0 , 0.0 , 1.0] ,  # Cropland
                3 : [0.78 , 0.78 , 0.39 , 1.0] ,  # Grassland
                4 : [0.0 , 0.78 , 0.0 , 1.0] ,  # Forest, broadleaf
                5 : [0.0 , 0.39 , 0.39 , 1.0] ,  # Forest, conifer
                6 : [0.0 , 0.0 , 1.0 , 1.0]}  # Water

    index_colors = [lcColors[key] if key in lcColors else (1 , 1 , 1 , 0) for key in range(1 , classification.max() + 1)]
    cmap = plt.matplotlib.colors.ListedColormap(index_colors , 'Classification' , classification.max())

    # prepare labels and patches for the legend
    labels = ['artificial land' , 'cropland' , 'grassland' , 'forest broadleaved' , 'forest coniferous' , 'water']
    patches = [patches.Patch(color=index_colors[i] , label=labels[i]) for i in range(len(labels))]

    # put those patched as legend-handles into the legend
    plt.legend(handles=patches , bbox_to_anchor=(1.05 , 1) , loc=2 , borderaxespad=0. , ncol=1 , frameon=False)
    plt.imshow(classification , cmap=cmap , interpolation='none')
    plt.show()

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