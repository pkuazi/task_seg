import numpy as np
import os, sys

from osgeo import gdal,ogr,osr
from skimage import exposure
from skimage.segmentation import felzenszwalb

import json
import gjsonc
from geotrans import GeomTrans

#BLOCK_SIZE=256
#OVERLAP_SIZE=13
BLOCK_SIZE=512
OVERLAP_SIZE=25
cluster_num=20

seg_path="/tmp/seg/"
if not os.path.exists(seg_path):
    os.makedirs(seg_path)
json_path='/tmp/json'
if not os.path.exists(json_path):
    os.makedirs(json_path)
encode_dir='/tmp/encode'
if not os.path.exists(encode_dir):
    os.makedirs(encode_dir)
   

def encode_json(t):
#     f=open(geojson_file)
#     t = json.loads(f.read())
    crs_wkt = t['crs']['properties']['name']
    num_geom=len(t['features'])
    for i in range(num_geom):
        print(i)
        geom = str(t['features'][i]['geometry'])
        geom_wgs = GeomTrans(crs_wkt, 'EPSG:4326').transform_json(geom)
        geojs=gjsonc.trunc_geojson(json.loads(geom_wgs),4)
        jstr = gjsonc.encode_geojson(geojs)
        t['features'][i]['geometry']=jstr
#         ruku(t['features'][i])
    return t     


def polygonize(src_filename,dst_filename):
    dst_layername = 'out'
    src_ds = gdal.Open( src_filename )
    if src_ds is None:
        print('Unable to open %s' % src_filename)
        sys.exit(1)
    srcband = src_ds.GetRasterBand(1)
    maskband = srcband.GetMaskBand()
    try:
        gdal.PushErrorHandler( 'QuietErrorHandler' )
        dst_ds = ogr.Open( dst_filename, update=1 )
        gdal.PopErrorHandler()
    except:
        dst_ds = None
  # =============================================================================
  #     Create output file.
  # =============================================================================
    if dst_ds is None:
        drv = ogr.GetDriverByName("GEOJSON")

    dst_ds = drv.CreateDataSource( dst_filename )

 # =============================================================================
#       Find or create destination layer.
 # =============================================================================
    try:
        dst_layer = dst_ds.GetLayerByName(dst_layername)
    except:
        dst_layer = None

    if dst_layer is None:
        srs = None
        if src_ds.GetProjectionRef() != '':
            srs = osr.SpatialReference()
            srs.ImportFromWkt( src_ds.GetProjectionRef() )
              
        dst_layer = dst_ds.CreateLayer(dst_layername, srs = srs )
             
        
        dst_fieldname = 'DN'
          
        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField(fd)
        dst_field = 0
    else:
        if dst_fieldname is not None:
            dst_field = dst_layer.GetLayerDefn().GetFieldIndex(dst_fieldname)
            if dst_field < 0:
                print("Warning: cannot find field '%s' in layer '%s'" % (dst_fieldname, dst_layername))


# =============================================================================
  #    Invoke algorithm.
 # =============================================================================

    prog_func = gdal.TermProgress
    
    result = gdal.Polygonize(srcband, maskband, dst_layer, dst_field, callback=prog_func)
    
    srcband = None
    src_ds = None
    dst_ds = None
    
    return dst_filename
        

def obia(band1,band2,band3, dst_file,proj, gt,jsonfile):

    bands_data = []
    bands_data.append(band1)
    bands_data.append(band2)
    bands_data.append(band3)

    bands_data = np.dstack(b for b in bands_data)
    if bands_data.min()==bands_data.max():
        print("The image do not need to segment")
        return None
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
        dst_ds=None
 
        vector_file = polygonize(dst_file,jsonfile)
#             cmd='gdal_polygonize.py %s -f GEOJSON  %s' % (dst_file,jsonfile)
#             os.system(cmd)
        return vector_file
    

def seg_raster(band1_file,band2_file,band3_file,imageid,seg_path):
    dataset1=gdal.Open(band1_file)
    if dataset1 is None:
        print("Failed to open file: " + band1_file)
        sys.exit(1)
    dataset2=gdal.Open(band2_file)
    if dataset2 is None:
        print("Failed to open file: " + band2_file)
        sys.exit(1)
    dataset3=gdal.Open(band3_file)
    if dataset3 is None:
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
            
            minx=gt[0]
            maxy=gt[3]
            maxx=gt[0]+BLOCK_SIZE*geotrans[1]
            miny=gt[3]+BLOCK_SIZE*geotrans[5]
            
            minx_wgs, maxy_wgs = GeomTrans(proj, 'EPSG:4326').transform_point([minx,maxy])
            maxx_wgs, miny_wgs = GeomTrans(proj, 'EPSG:4326').transform_point([maxx,miny])
            
            minx_list.append(minx_wgs)
            maxy_list.append(maxy_wgs)
            maxx_list.append(maxx_wgs)
            miny_list.append(miny_wgs)
        
            print("start segmenting...")
            jsonfile = os.path.join(json_path, imageid+str(i)+'_'+str(j)+'.geojson')
            json_file = obia(band1,band2,band3,dst_file,proj, gt,jsonfile)
            
            if json_file is None:
                continue
            else:
                f=open(json_file)
                geojson = json.loads(f.read())
                encode_gj= encode_json(geojson)
                dir_file=os.path.join(encode_dir, imageid+str(i)+'_'+str(j)+'.geojson')
                print(dir_file)    
                with open(dir_file, 'w') as outfile:
                    json.dump(encode_gj, outfile)
            
    bb = {}
    bb["subtaskname"]=subtask_list
    bb["minx"]=minx_list
    bb["maxy"]=maxy_list
    bb["maxx"]=maxx_list
    bb["miny"]=miny_list      

    df=pd.DataFrame(bb)
    df.to_csv('/tmp/subtask_bbox_wgs.csv')

if __name__ == '__main__':
    print('start')#RASTER_DATA_PATH = "/mnt/win/data/image/GF1/GF1_PMS1_E116.1_N40.0_20151012_L1A0001094174/"
    RASTER_DATA_PATH = "/mnt/win/data/BEIJING/l8_beijing/L8-OLI-123-032-20180408-LSR"
    imageid='L8-OLI-123-032-20180408-LSR'
    
    raster_b6 = os.path.join(RASTER_DATA_PATH, imageid+'-B6.TIF')
    raster_b5 = os.path.join(RASTER_DATA_PATH, imageid+'-B5.TIF')
    raster_b4 = os.path.join(RASTER_DATA_PATH, imageid+'-B4.TIF')
    
    seg_raster(raster_b6,raster_b5,raster_b4, imageid, seg_path )
    print("end")
