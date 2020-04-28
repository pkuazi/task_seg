#!/usr/bin/python
import json
import uuid

import psycopg2

import time

import os

from django.contrib.gis.geos import GEOSGeometry

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gscloud_web.settings")

import django

django.setup()

# 需要每次修改
taskid = "ea2e2c04c96f470aa41dae0299627b62"

path = "/root/encode_json"

box_path = "/root/subtask_bbox.csv"

conn = psycopg2.connect(database="gscloud_web", user="postgres", password="", host="10.0.81.19", port="9999")

print("Opened database successfully")

cur = conn.cursor()

labelid = "0"

ftype = "3"

state = "1"

userid = "9"

data_extent = {}


def gen_extent(file):
    '''
    生成数据边框
    :param file:
    :return:
    '''
    global data_extent
    with open(file, 'r') as fp:
        objs = fp.readlines()

    for obj in objs[1:]:
        data = obj.strip().split(",")
        filename = data[1]
        # x0, y0, x1, y1 = data[2], data[5], data[4], data[3]
        # extent = GEOSGeometry(
        #     "POLYGON (({x0} {y0}, {x0} {y1}, {x1} {y1}, {x1} {y0}, {x0} {y0}))".format(x0=x0,x1=x1,y0=y0,y1=y1))
        data_extent[filename] = data[2], data[5], data[4], data[3]



def gen_uuid():
    return str(uuid.uuid4()).replace("-", "")


def get_curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def createSubTask(files):
    '''
    创建子任务
    :param files:
    :return:
    '''
    sid = 1

    for file in files:
        filename = file.split(".geojson")[0]

        x0, y0, x1, y1 = data_extent[filename]

        boundary = GEOSGeometry(
            "POLYGON (({x0} {y0}, {x0} {y1}, {x1} {y1}, {x1} {y0}, {x0} {y0}))".format(x0=x0,x1=x1,y0=y0,y1=y1), srid=4326)

        guid = gen_uuid()

        print("--------------------------进行第{}个子任务 taskid:{}---------------------------".format(sid, guid))

        reward = 100

        ctime = get_curtime()

        sql = "insert into mark_subtask (guid,taskid,state,reward,ctime,mtime,sid,boundary,num_currentuser,num_maxuser,userlist) values('{guid}','{taskid}','{state}','{reward}','{ctime}','{mtime}','{sid}','{boundary}','{num_currentuser}','{num_maxuser}',array[''])".format(
            guid=guid, taskid=taskid, state=state, reward=reward, ctime=ctime, mtime=ctime, sid=sid,boundary = boundary, num_currentuser=0,
            num_maxuser=3)

        cur.execute(sql)

        conn.commit()
        #
        createSample(file, guid)

        sid += 1


def readFile(file):
    '''
    读取geojson文件
    :param file:
    :return:
    '''
    with open(file, 'r') as fp:
        data = fp.read()

    return data


def createSample(file, taskid):
    '''
    创建标注
    :param files:
    :return:
    '''

    filepath = path + "/" + file

    objs = json.loads(readFile(filepath))


    features = objs["features"]

    n = 1

    ctime = get_curtime()

    for feature in features:
        guid = gen_uuid()

        # geom = GEOSGeometry(json.dumps(feature["geometry"]))
        geojson = json.dumps(feature["geometry"])

        sql = "INSERT INTO mark_sample (guid,taskid,labelid,geojson,ftype,state,userid,ctime) VALUES ('{guid}','{taskid}','{labelid}','{geojson}','{ftype}','{state}','{userid}','{ctime}')".format(
            guid=guid, taskid=taskid, labelid=labelid, geojson=geojson, ftype=ftype, state=state, userid=userid,
            ctime=ctime)

        cur.execute(sql)
        if n % 100 == 0:
            print(n)

        n += 1
    conn.commit()


if __name__ == "__main__":
    start_time = time.time()
    gen_extent(box_path)

    files = os.listdir(path)
    #
    createSubTask(files)

    conn.close()

    print(time.time() - start_time)
