# ==================================================================================================== #
#   Assignment_11                                                                                      #
#   (c) Vincent Hanuschik, 07/01/2020                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import sklearn
import gdal
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# ======================================== SET TIME COUNT ============================================ #
time_start = time.localtime()
time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_start)
print("--------------------------------------------------------")
print("Start: " + time_start_str)
print("--------------------------------------------------------")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
dir = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Week11_Support-Vector-Machines/data/'
landsat = gdal.Open(dir+'landsat8_metrics1416.tif').ReadAsArray()
lucas = pd.read_csv(dir+ 'landsat8_metrics1416_samples.csv')


# Scitkilearn works on numpy arrays and pd dataframes

#SVM Hyperparameters: C - Cost; gamma - impact of kernel (defaultuse = rbf (radial something))
#grid search: create ranges for C and gamma and find best combination
    # should include cross validation methods

y = lucas['class']
X = lucas.drop(['class'], axis=1)
X = lucas.iloc[:, 1:]

from sklearn.preprocessing import StandardScaler
X_z = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_z, y, test_size=0.5, random_state= 42)

from sklearn.svm import SVC # "Support vector classifier"
model = SVC()
model.get_params()

from sklearn.model_selection import GridSearchCV    # GridSearch-CrossValidation
param_grid = {'C': [1, 5, 10, 50, 100, 1000],
              'gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid, cv=5, iid=False)
grid.fit(Xtrain, ytrain)
grid.best_estimator_.get_params()

from sklearn import metrics
svc_model = grid.best_estimator_
y_pred_svm = svc_model.predict(Xtest)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(ytest, y_pred_svm))
print("Confusion Matrix: ", "\n" , metrics.confusion_matrix(ytest, y_pred_svm))

# Classification report in text format
print(metrics.classification_report(ytest, y_pred_svm))

# apply model to standardized landsat image
ydim = landsat.shape[1]
xdim = landsat.shape[2]
landsat = landsat.transpose(1, 2, 0).reshape((ydim*xdim, landsat.shape[0]))

scaler = StandardScaler()
landsat_z = scaler.fit_transform(landsat)

classification = svc_model.predict(landsat_z)
rs = classification.reshape((ydim, xdim))

# quick map
plt.imshow(rs)
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

array2geotiff(rs, "SVM_classification.tif",ingdaxl=gdal.Open(dir+'landsat8_metrics1416.tif'))

# ======= ====== ====== plot map ====== ====== ======
from matplotlib import colors, patches

# plot land cover map
def plot(classification):
    lcColors = {1: [1.0, 0.0, 0.0, 1.0],  # Artificial
                2: [1.0, 1.0, 0.0, 1.0],  # Cropland
                3: [0.78, 0.78, 0.39, 1.0],  # Grassland
                4: [0.0, 0.78, 0.0, 1.0],  # Forest, broadleaf
                5: [0.0, 0.39, 0.39, 1.0],  # Forest, conifer
                6: [0.0, 0.0, 1.0, 1.0]}  # Water

    index_colors = [lcColors[key] if key in lcColors else (1, 1, 1, 0) for key in range(1, rs.max()+1)]
    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', rs.max())

    # prepare labels and patches for the legend
    labels = ['artificial land', 'cropland', 'grassland', 'forest broadleaved', 'forest coniferous', 'water']
    patches = [patches.Patch(color=index_colors[i], label=labels[i]) for i in range(len(labels))]

    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1, frameon=False)
    plt.imshow(rs, cmap=cmap, interpolation='none')
    plt.show()

# ======= ====== ====== write Geotiff to disk ====== ====== ======


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