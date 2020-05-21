# https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb

import numpy as np
import os, sys

from osgeo import gdal
from skimage import exposure
from skimage.segmentation import felzenszwalb


RASTER_DATA_PATH = "/mnt/win/data/image/l8_beijing/"
cluster_num=20
scale=90
sigma=0.1
min_size=10

def read_raster(rasterfile):
    dataset = gdal.OpenShared(rasterfile)
    if dataset is None:
        print("Failed to open file: " + rasterfile)
        sys.exit(1)
    band = dataset.GetRasterBand(1)
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    noDataValue = band.GetNoDataValue()

    rastervalue = band.ReadAsArray(xoff=0, yoff=0, win_xsize=xsize, win_ysize=ysize)
    rastervalue[rastervalue == noDataValue] = -9999

    return rastervalue, proj, geotrans

def obia_composite( cluster_num,rasterfile):
    dataset = gdal.Open(rasterfile)
    if dataset is None:
        print("Failed to open file: " + rasterfile)
        sys.exit(1)
    band = dataset.GetRasterBand(1)
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    gt=list(geotrans)
    noDataValue = band.GetNoDataValue()
    
    tile=dataset.ReadAsArray()
    print(tile.shape)
    band1=tile[3]
    band2=tile[1]
    band3=tile[2]
    band1[band1 == noDataValue] = -9999
    band2[band2 == noDataValue] = -9999
    band3[band3 == noDataValue] = -9999
    
    dst_file = "/tmp/classified.tif"

    bands_data = []
    
    bands_data.append(band1)
    
    bands_data.append(band2)
    
    bands_data.append(band3)

    bands_data = np.dstack(b for b in bands_data)
    img = exposure.rescale_intensity(bands_data)
    rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])

    segments = felzenszwalb(rgb_img, scale, sigma, min_size)
    print('finish segment')
    labels = np.unique(segments)
    X = []
    Y = []
    for l in labels:
        Y.append(l)
        mask = segments == l
        feature = []
        for i in range(3):
            img_b = img[:, :, i]
            feature.append(img_b[mask].mean())
        X.append(feature)
    from sklearn.cluster import KMeans
    x = np.array(X)
 
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(x)
    y_class_labels = kmeans.labels_
    for s in range(len(Y)):
        mask = segments == Y[s]
        segments[mask] = y_class_labels[s]

    xsize, ysize = segments.shape
    dst_format = 'GTiff'
    dst_nbands = 1
    dst_datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, dst_nbands, dst_datatype)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(segments)
    return dst_file

def obia( cluster_num):
    raster_b6 = os.path.join(RASTER_DATA_PATH, 'bj_b6_mask.tif')
    raster_b5 = os.path.join(RASTER_DATA_PATH, 'bj_b5_mask.tif')
    raster_b4 = os.path.join(RASTER_DATA_PATH, 'bj_b4_mask.tif')

    dst_file = "/tmp/test_classified.tif"

    bands_data = []
    band6, proj, gt = read_raster(raster_b6)
    bands_data.append(band6)
    band5, proj, gt = read_raster(raster_b5)
    bands_data.append(band5)
    band4, proj, gt = read_raster(raster_b4)
    bands_data.append(band4)

    bands_data = np.dstack(b for b in bands_data)
    img = exposure.rescale_intensity(bands_data)
    rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])

    segments = felzenszwalb(rgb_img, scale=85, sigma=0.25, min_size=9)

    labels = np.unique(segments)
    X = []
    Y = []
    for l in labels:
        Y.append(l)
        mask = segments == l
        feature = []
        for i in range(3):
            img_b = img[:, :, i]
            feature.append(img_b[mask].mean())
        X.append(feature)
    from sklearn.cluster import KMeans
    x = np.array(X)

    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(x)
    y_class_labels = kmeans.labels_
    for s in range(len(Y)):
        mask = segments == Y[s]
        segments[mask] = y_class_labels[s]

    xsize, ysize = segments.shape
    dst_format = 'GTiff'
    dst_nbands = 1
    dst_datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, dst_nbands, dst_datatype)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(segments)
    return dst_file


if __name__ == '__main__':
#     obia( cluster_num=20)
    data = '/root/Downloads/trt2013.img'
    obia_composite(20, data)
