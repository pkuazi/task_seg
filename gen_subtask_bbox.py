from osgeo import gdal, ogr
import os
from geotrans import GeomTrans

BLOCK_SIZE=512
OVERLAP_SIZE=0

def gen_subtask_bbox(rasterfile,imageid):
    print('the image is :', rasterfile)
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
                             
    bb = {}
    bb["subtaskname"]=subtask_list
    bb["minx"]=minx_list
    bb["maxy"]=maxy_list
    bb["maxx"]=maxx_list
    bb["miny"]=miny_list      
 
    df=pd.DataFrame(bb)
    df.to_csv('/tmp/subtask_512_bbox_wgs.csv')

if __name__ == '__main__':   
    task_data = '/mnt/win/data/sample_image/xiaoshan_2013.tif'
    gen_subtask_bbox(task_data,'xiaoshan_2013')