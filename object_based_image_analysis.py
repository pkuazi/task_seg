#https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb

import numpy as np
import os,sys
import scipy

from matplotlib import pyplot as plt
from matplotlib import colors
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift,felzenszwalb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

RASTER_DATA_PATH = "/mnt/win/image/l8_beijing/"
TRAIN_DATA_PATH = "/mnt/win/image/train"
TEST_DATA_PATH = "/mnt/win/image/test"

raster_b6 = os.path.join(RASTER_DATA_PATH,'bj_b6_mask.tif')
raster_b5 = os.path.join(RASTER_DATA_PATH,'bj_b5_mask.tif')
raster_b4 = os.path.join(RASTER_DATA_PATH,'bj_b4_mask.tif')

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
    print("Output file %s size:" % rasterfile, xsize, ysize)
    rastervalue = band.ReadAsArray(xoff=0, yoff=0, win_xsize=xsize, win_ysize=ysize)
    rastervalue[rastervalue == noDataValue] = -9999

    return rastervalue


bands_data = []
band6=read_raster(raster_b6)
bands_data.append(band6)
band5 = read_raster(raster_b5)
bands_data.append(band5)
band4 = read_raster(raster_b4)
bands_data.append(band4)

bands_data = np.dstack(b for b in bands_data)
img = exposure.rescale_intensity(bands_data)
rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
plt.figure()
plt.imshow(rgb_img)
plt.show()
#
# # Segments image using quickshift clustering in Color-(x,y) space.
# # Produces an oversegmentation of the image using the quickshift mode-seeking algorithm.
segments_quick = quickshift(img, kernel_size=7, max_dist=3, ratio=0.35, convert2lab=False)
n_segments = len(np.unique(segments_quick))
print(n_segments)

cmap = colors.ListedColormap(np.random.rand(n_segments, 3))
plt.figure()
plt.imshow(segments_quick, interpolation='none', cmap=cmap)
plt.show()

band_segmentation = []
for i in range(3):
    band_segmentation.append(felzenszwalb(img[:, :, i], scale=85, sigma=0.25, min_size=9))

const = [b.max() + 1 for b in band_segmentation]
segmentation = band_segmentation[0]
for i, s in enumerate(band_segmentation[1:]):
    segmentation += s * np.prod(const[:i+1])

_, labels = np.unique(segmentation, return_inverse=True)
segments_felz = labels.reshape(img.shape[:2])

cmap = colors.ListedColormap(np.random.rand(len(np.unique(segments_felz)), 3))
plt.figure()
plt.imshow(segments_felz, interpolation='none', cmap=cmap)

n_segments = max(len(np.unique(s)) for s in [segments_quick, segments_felz])

cmap = colors.ListedColormap(np.random.rand(n_segments, 3))
#SHOW_IMAGES:
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(rgb_img, interpolation='none')
ax1.set_title('Original image')
ax2.imshow(segments_quick, interpolation='none', cmap=cmap)
ax2.set_title('Quickshift segmentations')
ax3.imshow(segments_felz, interpolation='none', cmap=cmap)
ax3.set_title('Felzenszwalb segmentations')
plt.show()

