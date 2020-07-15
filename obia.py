# https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb

import numpy as np
import os, sys

from osgeo import gdal
from skimage import exposure
from skimage.segmentation import felzenszwalb
from geotrans import GeomTrans

RASTER_DATA_PATH = "/mnt/win/data/image/l8_beijing/"
cluster_num=20
scale=90
sigma=0.1
min_size=10

def read_raster_by_region(rasterfile,bbox):
    dataset = gdal.OpenShared(rasterfile)
    if dataset is None:
        print("Failed to open file: " + rasterfile)
        sys.exit(1)
    band = dataset.GetRasterBand(1)
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    proj = dataset.GetProjection()
    gt = dataset.GetGeoTransform()
    noDataValue = band.GetNoDataValue()
    
    geotrans=list(gt)
    
    minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    
    minx_img, maxy_img = GeomTrans( 'EPSG:4326',proj).transform_point([minx,maxy])
    maxx_img, miny_img = GeomTrans('EPSG:4326',proj).transform_point([maxx,miny])
    
    xoff = round((minx_img-gt[0])/gt[1])
    yoff = round((maxy_img-gt[3])/gt[5])
    
    geotrans[0] = gt[0]+xoff*gt[1]
    geotrans[3] = gt[3]+yoff*gt[5]
    
    xsize = int((maxx_img-minx_img)/gt[1])+1
    ysize = int((miny_img-maxy_img)/gt[5])+1

    rastervalue = band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
#     rastervalue[rastervalue == noDataValue] = -9999

    return rastervalue, proj, geotrans

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

def obia( cluster_num, raster_b6, raster_b5, raster_b4):
#     raster_b6 = os.path.join(RASTER_DATA_PATH, 'bj_b6_mask.tif')
#     raster_b5 = os.path.join(RASTER_DATA_PATH, 'bj_b5_mask.tif')
#     raster_b4 = os.path.join(RASTER_DATA_PATH, 'bj_b4_mask.tif')

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
    
def obia_by_region( cluster_num, raster_b6, raster_b5, raster_b4,bbox, dst_file):
    bands_data = []
    band6, proj, gt = read_raster_by_region(raster_b6, bbox)
    bands_data.append(band6)
    band5, proj, gt = read_raster_by_region(raster_b5, bbox)
    bands_data.append(band5)
    band4, proj, gt = read_raster_by_region(raster_b4, bbox)
    bands_data.append(band4)

    bands_data = np.dstack(b for b in bands_data)
    img = exposure.rescale_intensity(bands_data)
    rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
    
    print('begin segmenting...')
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
    
    print('begin classifying...')
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
    print('finished...')
    return dst_file

if __name__ == '__main__':
#     obia( cluster_num=20)
    xiongan_bbox = [115.641442, 116.335359, 38.727196, 39.164298]
#     image_path = '/mnt/win/data/xiongan/images/LC81230332017191LGN00'
    image_path = '/mnt/win/data/xiongan/images/LC81230332017271LGN00'
    files = os.listdir(image_path)
    raster_b4=''
    raster_b5=''
    raster_b6=''
    for file in files:
        if file.endswith('B6.TIF'):
            raster_b6 = os.path.join(image_path, file)
        elif file.endswith('B5.TIF'):
            raster_b5 = os.path.join(image_path, file)
        elif file.endswith('B4.TIF'):
            raster_b4 = os.path.join(image_path, file)
    if raster_b4!='' and raster_b5!='' and raster_b6!='':
        dst_file = "/tmp/test271_classified.tif"
#         obia_by_region(20, raster_b6,raster_b5,raster_b4,xiongan_bbox,dst_file)
        dst_json = '/tmp/test271_classifiedss.geojson'
        cmd='gdal_polygonize.py %s -f GEOJSON  %s' % (dst_file,dst_json)
        os.system(cmd)
        
        
#     data = '/root/Downloads/trt2013.img'
#     obia_composite(20, data)
