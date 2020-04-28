import numpy as np
import os, sys

from osgeo import gdal
from skimage import exposure
from skimage.segmentation import felzenszwalb

import json
import gjsonc


#RASTER_DATA_PATH = "/mnt/win/data/image/GF1/GF1_PMS1_E116.1_N40.0_20151012_L1A0001094174/"
RASTER_DATA_PATH = "/root"
#image="GF1_PMS1_E116.1_N40.0_20151012_L1A0001094174-MSS1.tiff"
image="GF1_PMS1_E116.1_N40.0.tif"
imge_path=os.path.join(RASTER_DATA_PATH,image)
seg_path="/tmp/seg/"

#BLOCK_SIZE=256
#OVERLAP_SIZE=13
BLOCK_SIZE=512
OVERLAP_SIZE=25
cluster_num=64

def obia(band1,band2,band3, dst_file,proj, gt):

    bands_data = []
    bands_data.append(band1)
    bands_data.append(band2)
    bands_data.append(band3)

    bands_data = np.dstack(b for b in bands_data)
    if bands_data.min()==bands_data.max():
        print("The image %s do not need to segment"%(image))
    else:
        img = exposure.rescale_intensity(bands_data)
        rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
        print("tile size: ",rgb_img.shape)
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
        print(x.shape)
        try:
            kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(x)
            y_class_labels = kmeans.labels_
            for s in range(len(Y)):
                mask = segments == Y[s]
                segments[mask] = y_class_labels[s]
        except ValueError as e:
            print("sample number is lower than cluster num")

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

def ruku(record):

def encode_json(geojson_file):
    f=open(geojson_file)
    t = json.loads(f.read())
    num_geom=len(t['features'])
    for i in range(num_geom):
        print(i)
        geom = t['features'][i]['geometry']
        geojs=gjsonc.trunc_geojson(geom,4)
        jstr = gjsonc.encode_geojson(geojs)
        t['features'][i]['geometry']=jstr
		ruku(t['features'][i])
        
    dir_file=os.path.join(encode_dir, files[n])
    # print(dir_file)    
    # with open(dir_file, 'w') as outfile:
        # json.dump(t, outfile)

def seg_raster(band1_file,band2_file,band3_file,imageid,seg_path):
    dataset1=gdal.Open(band1_file)
    if dataset is None:
        print("Failed to open file: " + band1_file)
        sys.exit(1)
    dataset1=gdal.Open(band2_file)
    if dataset is None:
        print("Failed to open file: " + band2_file)
        sys.exit(1)
    dataset1=gdal.Open(band3_file)
    if dataset is None:
        print("Failed to open file: " + band3_file)
        sys.exit(1)
    band = dataset1.GetRasterBand(1)
    xsize = dataset1.RasterXSize
    ysize = dataset1.RasterYSize
    proj = dataset1.GetProjection()
    geotrans = dataset1.GetGeoTransform()
    gt=list(geotrans)
    noDataValue = band.GetNoDataValue()
    
    rnum_tile=int((ysize-BLOCK_SIZE)/(BLOCK_SIZE-OVERLAP_SIZE))+1
    cnum_tile=int((xsize-BLOCK_SIZE)/(BLOCK_SIZE-OVERLAP_SIZE))+1
    print('the number of tile is :',rnum_tile*cnum_tile)
    
    import pandas as pd
    subtask_list=[]
    minx_list=[]
    maxy_list=[]
    maxx_list=[]
    miny_list=[]
    
    for i in range(rnum_tile):
        print(i)
        for j in range(cnum_tile):
            xoff=0+(BLOCK_SIZE-OVERLAP_SIZE)*j
            yoff=0+(BLOCK_SIZE-OVERLAP_SIZE)*i
            print("the row and column of tile is :", xoff, yoff)
            tile=dataset.ReadAsArray(xoff, yoff, BLOCK_SIZE,BLOCK_SIZE)
            print(tile.shape)
            band1=dataset1.ReadAsArray(xoff, yoff, BLOCK_SIZE,BLOCK_SIZE)
            band2=dataset2.ReadAsArray(xoff, yoff, BLOCK_SIZE,BLOCK_SIZE)
            band3=dataset3.ReadAsArray(xoff, yoff, BLOCK_SIZE,BLOCK_SIZE)
            band1[band1 == noDataValue] = -9999
            band2[band2 == noDataValue] = -9999
            band3[band3 == noDataValue] = -9999

            dst_file=os.path.join(seg_path,imageid+str(i)+'_'+str(j)+'.tif')
            gt[0]=geotrans[0]+xoff*geotrans[1]
            gt[3]=geotrans[3]+yoff*geotrans[5]
            
            subtask_list.append(imageid+str(i)+'_'+str(j))
            minx_list.append(gt[0])
            maxy_list.append(gt[3])
            maxx_list.append(gt[0]+BLOCK_SIZE*geotrans[1])
            miny_list.append(gt[3]+BLOCK_SIZE*geotrans[5])
        
            print("start segmenting...")
            dst_file=obia(band1,band2,band3,dst_file,proj, gt)
            
            
            jsonfile = os.path.join(json_path, imageid+str(i)+'_'+str(j)+'.geojson')
            cmd='gdal_polygonize.py %s -f GEOJSON  %s' % (dst_file,jsonfile)
            os.system(cmd)
            
            encode_json(jsonfile)
            
    bb = {}
    bb["subtaskname"]=subtask_list
    bb["minx"]=minx_list
    bb["maxy"]=maxy_list
    bb["maxx"]=maxx_list
    bb["miny"]=miny_list      
    #存储到pandas的dataframe中（numpy的array不能存储不同类型，如字符串和数字）
    df=pd.DataFrame(bb)
    df.to_csv('subtask_bbox.csv')

if __name__ == '__main__':
    print('start')
    print('the image is :', rasterfile)
    RASTER_DATA_PATH='/mnt/win/data/BEIJING/bj_l8/L8-OLI-123-032-20180408-LSR'
    imageid='L8-OLI-123-032-20180408-LSR'
    raster_b6 = os.path.join(RASTER_DATA_PATH, imageid+'-B6.TIF')
    raster_b5 = os.path.join(RASTER_DATA_PATH, imageid+'-B5.TIF')
    raster_b4 = os.path.join(RASTER_DATA_PATH, imageid+'-B4.TIF'')
    
    seg_path='/tmp'
    seg_raster(raster_b6,raster_b5,raster_b4, imageid, seg_path )
    print("end")
