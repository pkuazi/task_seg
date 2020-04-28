# https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb

import numpy as np
import os, sys, re
import scipy
import rasterio

import numpy.ma as ma

from matplotlib import pyplot as plt
from matplotlib import colors
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, felzenszwalb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import ogr

RASTER_DATA_PATH = "/mnt/win/image/l8_beijing/"


def wkt_from_shp(shpfile):
    dr = ogr.GetDriverByName("ESRI Shapefile")
    ds = dr.Open(shpfile)
    layer = ds.GetLayer(0)
    extent = layer.GetExtent()  # minx, maxx, miny,  maxy
    feat = layer.GetFeature(0)
    geom = feat.GetGeometryRef()
    return geom.ExportToWkt()


def extent_from_wkt(wkt):
    nums = re.findall(r'\d+(?:\.\d*)?', wkt.rpartition(',')[0])
    coords_str = zip(*[iter(nums)] * 2)
    coors_int = np.array(coords_str, np.int32)
    xmin = coors_int[:, 0].min()
    xmax = coors_int[:, 0].max()
    ymin = coors_int[:, 1].min()
    ymax = coors_int[:, 1].max()
    return [xmin, xmax, ymin, ymax]


def raster_read(wkt, raster_file):
    mask_bounds = extent_from_wkt(wkt)

    minx = mask_bounds[0]
    maxx = mask_bounds[1]
    miny = mask_bounds[2]
    maxy = mask_bounds[3]
    ds = rasterio.open(raster_file, 'r')
    proj = str(ds.crs.wkt)
    transform = ds.transform

    mask_transform = [minx, transform[1], transform[2], maxy, transform[4], transform[5]]
    window = ds.window(*mask_bounds)

    (res_width, res_height) = ds.res
    x_start = window[0][0]
    x_end = x_start + int((maxy - miny) / res_height)
    y_start = window[1][0]
    y_end = y_start + int((maxx - minx) / res_width)

    if x_end > ds.width:
        x_end = ds.width
    if y_end > ds.height:
        y_end = ds.height

    out_image = ds.read()[0][x_start:x_end, y_start:y_end]
    # out_image = ds.read(window=window, masked=False)
    return out_image, proj, mask_transform


def arr2rst(arr, dst_file, proj, geotransform):
    # output the array in geotiff format
    xsize, ysize = arr.shape
    dst_format = 'GTiff'
    dst_nbands = 1
    # dst_datatype = gdal.GDT_UInt32
    dst_datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, dst_nbands, dst_datatype)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(proj)
    # dst_ds.GetRasterBand(1).SetNoDataValue(params[2])
    dst_ds.GetRasterBand(1).WriteArray(arr)
    # return dst_file


def felzenszwalb_segment(img):
    band_segmentation = []

    for i in range(3):
        band_segmentation.append(felzenszwalb(img[:, :, i], scale=49, sigma=0.25, min_size=9))

    const = [b.max() + 1 for b in band_segmentation]
    segmentation = band_segmentation[0]

    for i, s in enumerate(band_segmentation[1:]):
        segmentation += s * np.prod(const[:i + 1])

    _, labels = np.unique(segmentation, return_inverse=True)

    segments_felz = labels.reshape(img.shape[:2])
    return segments_felz


def segments_classify(img, segments):
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
    cluster_num = 10
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(x)
    y_class_labels =  kmeans.labels_
    for s in range(len(Y)):
        mask = segments==Y[s]
        segments[mask]=y_class_labels[s]
    return segments



def main():
    # raster_b6 = os.path.join(RASTER_DATA_PATH, 'bj_b6_mask.tif')
    # raster_b5 = os.path.join(RASTER_DATA_PATH, 'bj_b5_mask.tif')
    # raster_b4 = os.path.join(RASTER_DATA_PATH, 'bj_b4_mask.tif')

    shpfile = '/mnt/win/patent/transform_grid_12432_591_261.shp'
    wkt = wkt_from_shp(shpfile)

    raster_b6 = '/mnt/win/image/LC81240322014270LGN00_B6.TIF'
    raster_b5 = '/mnt/win/image/LC81240322014270LGN00_B5.TIF'
    raster_b4 = '/mnt/win/image/LC81240322014270LGN00_B4.TIF'

    bands_data = []
    band6, proj, mask_transform = raster_read(wkt, raster_b6)
    bands_data.append(band6)
    band5, proj, mask_transform = raster_read(wkt, raster_b5)
    bands_data.append(band5)
    band4, proj, mask_transform = raster_read(wkt, raster_b4)
    bands_data.append(band4)

    bands_data = np.dstack(b for b in bands_data)
    img = exposure.rescale_intensity(bands_data)
    # rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])

    import time
    t = time.time()
    segments_felz = felzenszwalb_segment(img)
    t = time.time() - t
    print t
    arr2rst(segments_felz, '/tmp/f7_seg49.tif', proj, mask_transform)

    segments_class = segments_classify(img, segments_felz)
    import time
    t = time.time() - t
    print t
    # save to geotiff
    arr2rst(segments_class, '/tmp/f2_classified.tif', proj, mask_transform)

    # cmap = colors.ListedColormap(np.random.rand(len(np.unique(segments_felz)), 3))
    # plt.figure()
    # plt.imshow(segments_felz, interpolation='none', cmap=cmap)


if __name__ == '__main__':
    main()

    # test
    # shpfile = '/tmp/grid_12432_589_260.shp'
    # wkt = wkt_from_shp(shpfile)
    # raster_file = '/mnt/win/image/LC81240322014270LGN00_B7.TIF'
    # arr, proj, mask_transform = raster_read(wkt, raster_file)
    # arr2rst(arr, '/tmp/mask12431_589_260.tif', proj, mask_transform)
