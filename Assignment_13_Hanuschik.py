# ==================================================================================================== #
#   Assignment_13                                                                                      #
#   (c) Vincent Hanuschik, 21/01/2020                                                                  #
#                                                                                                      #
# ================================== LOAD REQUIRED LIBRARIES ========================================= #
import time
import os
import numpy as np
from datetime import datetime
import gdal
from matplotlib import pyplot as plt

# ======================================== SET TIME COUNT ============================================ #
time_start = time.localtime()
time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S", time_start)
print("--------------------------------------------------------")
print("Start: " + time_start_str)
print("--------------------------------------------------------")
# =================================== DATA PATHS AND DIRECTORIES====================================== #
#   * unpacks list in function argument
# fun(*mylist)

#   ** unpacks dictionary in function argument
# fun(**mydict)

# def fun(a1,b1,c1):
#     print(a1,b1,c1)
# d = {'a1':2, 'c1':4, 'b1':10}
# fun(**d)
# lstsq = least squares

dir = '/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Week13 - Time series/'
evi_file = 'data/evi_X0070_Y0040_subset.bsq'

def plotts(t, vi, fit = None):

    ts = plt.plot(t, vi, 'ko', fillstyle='none')
    if fit is not None:
        t_fit, vi_fit = fit
        plt.plot(t_fit, vi_fit, 'blue')

    plt.xlabel('Time (days)')
    plt.ylabel('EVI')
    plt.ylim(0, 1)

src_ds = gdal.Open(dir+evi_file)
bandNames = [src_ds.GetRasterBand(i).GetDescription() for i in range(1, src_ds.RasterCount + 1)]
tstamps = np.array([datetime.strptime(x, '%Y-%m-%d') for x in bandNames])
doys = np.array([x.timetuple().tm_yday for x in tstamps])
ddoys = np.array([(x - datetime(2016, 1, 1)).days + 1 for x in tstamps])
years = np.array([x.timetuple().tm_year for x in tstamps])
dyears = np.array([x.timetuple().tm_year + x.timetuple().tm_yday/datetime(x.timetuple().tm_year, 12, 31).timetuple().tm_yday for x in tstamps])

i = np.logical_and(years > 2015, years < 2020)
evi = src_ds.ReadAsArray()[i, :, :]
nodata = evi == -32767
evi = evi / 10000.0
evi[nodata] = np.nan
doys = doys[i]
tstamps = tstamps[i]

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forest I ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
y = evi[:, 300, 100]
m = np.isnan(y) == False
y_f = y[m]
t_doys = doys[m]
t = tstamps[m]
t = dyears[m]
t_f = ddoys[m]

period = 365
x_f = t_f
A = np.array([np.ones_like(x_f), x_f, x_f * x_f,
              np.cos(2.0 * np.pi * (x_f / period)),
              np.sin(2.0 * np.pi * (x_f / period))])


def trend_with_harmonic(t, a0, a1, a2, a3, a4, period=365):
    result = a0 + a1*t + a2*t*t + \
             a3 * np.cos(2 * np.pi * (t / period)) + \
             a4 * np.sin(2 * np.pi * (t / period))
    return result

# Call lstsq
popt, sum_of_residuals, r, evals = np.linalg.lstsq(A.T, y_f, rcond=None)
y_fit_harmonic_forest = trend_with_harmonic(x_f, *popt)

plt.figure(figsize=(12, 7))
plt.plot(t_f, y_fit_harmonic_forest, '-', lw=1, label="Fitted")
plotts(t_f, y_f)
plt.xlabel("Time"); plt.ylabel("EVI"); plt.legend(loc="best")
plt.show()

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forest II ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
A = np.array([np.ones_like(x_f), x_f, x_f * x_f,
              np.cos(2.0 * np.pi * (x_f / period)),
              np.sin(2.0 * np.pi * (x_f / period)),
              np.cos(2.0 * 2.0 * np.pi * (x_f / period)),
              np.sin(2.0 * 2.0 * np.pi * (x_f / period))])

popt_f, sum_of_residuals_f, r_f, evals_f = np.linalg.lstsq(A.T, y_f, rcond=None)

def trend_with_harmonic_xtnd(t, a0, a1, a2, a3, a4, a5, a6, period=365):
    result = a0 + a1*t + a2*t*t + \
             a3 * np.cos(2 * np.pi * (t / period)) + \
             a4 * np.sin(2 * np.pi * (t / period)) + \
             a5 * np.cos(2 * 2 * np.pi * (t / period)) + \
             a6 * np.sin(2 * 2 * np.pi * (t / period))
    return result

y_fit_harmonic_forestII = trend_with_harmonic_xtnd(x_f, *popt_f)

plt.figure(figsize=(12, 7))
plt.plot(t_f, y_fit_harmonic_forestII, '-', lw=1, label="Fitted")
plotts(t_f, y_f)
plt.xlabel("Time"); plt.ylabel("EVI"); plt.legend(loc="best")
plt.show()

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Crop I ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #
y = evi[:, 207, 100]
m = np.isnan(y) == False
y_c = y[m]
t_doys = doys[m]
t = tstamps[m]
t = dyears[m]
t_c = ddoys[m]
# plot_ts(t, y)

# fit harmonic functions
period = 365
x_c = t_c
A = np.array([np.ones_like(x_c), x_c, x_c * x_c,
              np.cos(2.0 * np.pi * (x_c / period)),
              np.sin(2.0 * np.pi * (x_c / period))])

# Call lstsq
popt_c, sum_of_residuals_c, r_c, evals_c = np.linalg.lstsq(A.T, y_c, rcond=None)
y_fit_harmonic_cropland = trend_with_harmonic(x_c, *popt_c)

plt.figure(figsize=(12, 7))
plt.plot(t_c, y_fit_harmonic_cropland, '-', lw=1, label="Fitted")
plotts(t_c, y_c)
plt.xlabel("Time"); plt.ylabel("EVI"); plt.legend(loc="best")
plt.show()

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Crop II ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #

A = np.array([np.ones_like(x_c), x_c, x_c * x_c,
              np.cos(2 * np.pi * (x_c / period)),
              np.sin(2 * np.pi * (x_c / period)),
              np.sin(2 * 2 * np.pi * (x_c / period)),
              np.sin(2 * 2 * np.pi * (x_c / period))])

popt_c, sum_of_residuals_c, r_c, evals_c = np.linalg.lstsq(A.T, y_c, rcond=None)

def trend_with_harmonic_xtnd_crop(t, a0, a1, a2, a3, a4, a5, a6, period=365):
    result = a0 + a1 * t + a2 * t * t + \
             a3 * np.cos(2 * np.pi * (t / period)) + \
             a4 * np.sin(2 * np.pi * (t / period)) + \
             a5 * np.sin(2 * 2 * np.pi * (t / period)) + \
             a6 * np.sin(2 * 2 * np.pi * (t / period))
    return result

y_fit_harmonic_cropII = trend_with_harmonic_xtnd_crop(x_c, *popt_c)

plt.figure(figsize=(12, 7))
plt.plot(t_c, y_fit_harmonic_cropII, '-', lw=1, label="Fitted")
plotts(t_c, y_c)
plt.xlabel("Time"); plt.ylabel("EVI"); plt.legend(loc="best")
plt.show()

# ==================================================================================================== #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exercise III ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ==================================================================================================== #


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