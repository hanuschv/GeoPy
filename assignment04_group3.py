import os
import gdal

path = "/Users/Vince/Documents/Uni MSc/Msc 7 Geoprocessing with Python/Assignment04_Files/"
image = gdal.Open(path + 'tileID_410_y2018.tif')

gt = image.GetGeoTransform()
print (str('real geo coordinates:'), gt)

inv_gt = gdal.InvGeoTransform(gt)
print (str('array coordinates:'), inv_gt)
print(str('If all went well then the success flag will be 1 but if the affine transformation couldnot be inverted it returns 0 instead'))

## upper left to get the offset distance x, y
app_gt_offset_upperleft = gdal.ApplyGeoTransform(inv_gt, gt[0], gt[1]) #common extent upper-left-corner
print (app_gt_offset_upperleft)
app_gt_offset_lowerright = gdal.ApplyGeoTransform(inv_gt, -63.4964277112, -24.8040121305) #common extent lower-right-corner
print (app_gt_offset_lowerright)

#sclicing = offset + size of smaller extent
#array index needs to be an integer
off_ulx, off_uly = map(int, app_gt_offset_upperleft)
print (str("off_ulx = "), off_ulx, str("| off_uly = "), off_uly)
off_lrx, off_lry = map(int, app_gt_offset_lowerright)
print (str("off_lrx = "), off_lrx, str("| off_lry = "), off_lry)
## substract to get row & col distance (number)
rows = off_lry - off_uly
columns = off_lrx - off_ulx

print (str("row = "), rows, str("| col = "), columns)
## image crop = upperleftx, upperlefty, row, colums
value = image.ReadAsArray(off_ulx, off_uly, 1326, 1272)
print (str("resulting array from the common extent:"), value)

import numpy as np
print (value.mean())