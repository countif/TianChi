from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
import time
import operator
from pandas import DataFrame
import csv
import math
import random
from copy import deepcopy
from sklearn.utils import shuffle
from datetime import datetime,date

train_path = "../../data/Split/full/train.csv"
test_path = "../../data/Split/full/test.csv"

#train_path = "../../data/deal/split/train.csv"

#test_path = "../../data/eval/split/eval9.csv"

shop_path = "/home/yuanbin/tianchi/data/shop/shop_info.csv"


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def rad(d):
    return d*3.1415926/180


def ComputeDistance(ra1,lg1,ra2,lg2):
    rad1 = rad(ra1)
    rad2 = rad(ra2)

    a = rad1 - rad2
    b = rad(lg1) - rad(lg2)

    # distance = dx*dx + dy*dy
    distance = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(rad1) * math.cos(rad2) * math.pow(math.sin(b / 2), 2)))
    distance = distance*6371004
    return distance


def BuildShopDicWifi(train):
    ShopDicWifi = {}
    WifiConnect = {}
    WifiConnectShop = {}
    WifiAvgSignalShop = {}
    WifiAvgRankShop = {}
    WifiDicShopFirst = {}
    WifiDicShopSecond = {}
    WifiAvgSignalShopFirst = {}
    WifiAvgSignalShopSecond = {}

    WifiDicShop = {}

    WifiMedianSignalShop = {}
    WifiConnectSingalShop = {}

    values = train['wifi_infos'].values
    Shops = train['shop_id'].values
    for shop in Shops:
        ShopDicWifi[shop] = {}

    ShopCnt = {}
    for shop in Shops:
        if shop in ShopCnt:
            ShopCnt[shop]  = ShopCnt[shop] + 1
        else:
            ShopCnt[shop] = 1

    Len = len(values)

    result = []
    for i in  range(0,Len):
        tokens = values[i].split(";")
        dic = {}
        for x in tokens:
            xr = x.split("|")
            if xr[2] == 'true':
                if xr[0] in WifiConnect:
                    WifiConnect[xr[0]] = WifiConnect[xr[0]]
                else:
                    WifiConnect[xr[0]] = 1
                if xr[0] in WifiConnectShop:
                    if Shops[i] in WifiConnectShop[xr[0]]:
                        WifiConnectShop[xr[0]][Shops[i]] = WifiConnectShop[xr[0]][Shops[i]] + 1
                    else:
                        WifiConnectShop[xr[0]][Shops[i]] = 1
                else:
                    WifiConnectShop[xr[0]] = {}
                    WifiConnectShop[xr[0]][Shops[i]] = 1
                if xr[0] in WifiConnectSingalShop:
                    if Shops[i] in WifiConnectSingalShop[xr[0]]:
                        WifiConnectSingalShop[xr[0]][Shops[i]].append(float(xr[1]))
                    else:
                        WifiConnectSingalShop[xr[0]][Shops[i]] = []
                        WifiConnectSingalShop[xr[0]][Shops[i]].append(float(xr[1]))
                else:
                    WifiConnectSingalShop[xr[0]] = {}
                    WifiConnectSingalShop[xr[0]][Shops[i]] = []
                    WifiConnectSingalShop[xr[0]][Shops[i]].append(float(xr[1]))

            dic[xr[0]] = float(xr[1])
            if xr[0] in WifiAvgSignalShop:
                if (Shops[i] in WifiAvgSignalShop[xr[0]]):
                    WifiAvgSignalShop[xr[0]][Shops[i]].append(float(xr[1]))
                else:
                    WifiAvgSignalShop[xr[0]][Shops[i]]= []
                    WifiAvgSignalShop[xr[0]][Shops[i]].append(float(xr[1]))
            else:
                WifiAvgSignalShop[xr[0]] = {}
                WifiAvgSignalShop[xr[0]][Shops[i]] = []
                WifiAvgSignalShop[xr[0]][Shops[i]].append(float(xr[1]))

        result.append(dic)

    for wifi in WifiAvgSignalShop:
        WifiMedianSignalShop[wifi] = {}
        for shop in WifiAvgSignalShop[wifi]:
            arr = WifiAvgSignalShop[wifi][shop]
            arr.sort()
            median = arr[len(arr)/2]
            WifiMedianSignalShop[wifi][shop] = median

    for wifi in WifiAvgSignalShop:
        for shop in WifiAvgSignalShop[wifi]:
            arr = WifiAvgSignalShop[wifi][shop]
            sum = 0.0
            for i in range(0,len(arr)):
                sum = sum + arr[i]
            sum = sum / len(arr)
            WifiAvgSignalShop[wifi][shop] = sum

    for wifi in WifiConnectSingalShop:
        for shop in WifiConnectSingalShop[wifi]:
            arr = WifiConnectSingalShop[wifi][shop]
            sum = 0.0
            for i in range(0,len(arr)):
                sum = sum + arr[i]
            sum = sum / len(arr)
            WifiConnectSingalShop[wifi][shop] = sum

    for i in range(0,Len):
        arr = []
        cur = result[i]
        shop_id =Shops[i]

        for wifi in cur:
            arr.append([wifi,cur[wifi]])
        arr.sort(lambda x, y: cmp(x[1], y[1]),reverse=True)



        for j in range(0,len(arr)):
            wifi = arr[j][0]
            rank = j
            if wifi in WifiAvgSignalShopFirst:
                if j < 2:
                    if (Shops[i] in WifiAvgSignalShopFirst[wifi]):
                        WifiAvgSignalShopFirst[wifi][Shops[i]].append(float(arr[j][1]))
                    else:
                        WifiAvgSignalShopFirst[wifi][Shops[i]] = []
                        WifiAvgSignalShopFirst[wifi][Shops[i]].append(float(arr[j][1]))
            else:
                if j < 2:
                    WifiAvgSignalShopFirst[wifi] = {}
                    WifiAvgSignalShopFirst[wifi][Shops[i]] = []
                    WifiAvgSignalShopFirst[wifi][Shops[i]].append(float(arr[j][1]))

            if wifi in WifiAvgSignalShopSecond:
                if j > 2:
                    if (Shops[i] in WifiAvgSignalShopSecond[wifi]):
                        WifiAvgSignalShopSecond[wifi][Shops[i]].append(float(arr[j][1]))
                    else:
                        WifiAvgSignalShopSecond[wifi][Shops[i]] = []
                        WifiAvgSignalShopSecond[wifi][Shops[i]].append(float(arr[j][1]))
            else:
                if j > 2:
                    WifiAvgSignalShopSecond[wifi] = {}
                    WifiAvgSignalShopSecond[wifi][Shops[i]] = []
                    WifiAvgSignalShopSecond[wifi][Shops[i]].append(float(arr[j][1]))

            if wifi in WifiDicShopFirst:
                if j <2:
                    if (Shops[i] in WifiDicShopFirst[wifi]):
                        WifiDicShopFirst[wifi][Shops[i]] = WifiDicShopFirst[wifi][Shops[i]] + 1
                    else:
                        WifiDicShopFirst[wifi][Shops[i]]= 1.0
            else:
                if j < 2:
                    WifiDicShopFirst[wifi] = {}
                    WifiDicShopFirst[wifi][Shops[i]] = 1.0

            if wifi in WifiDicShopSecond:
                if j < 2:
                    if (Shops[i] in WifiDicShopSecond[wifi]):
                        WifiDicShopSecond[wifi][Shops[i]] = WifiDicShopSecond[wifi][Shops[i]]  + 1
                    else:
                        WifiDicShopSecond[wifi][Shops[i]]= 1.0
            else:
                if j < 2:
                    WifiDicShopSecond[wifi] = {}
                    WifiDicShopSecond[wifi][Shops[i]] = 1.0

            if wifi in WifiAvgRankShop:
                if (Shops[i] in WifiAvgRankShop[wifi]):
                    WifiAvgRankShop[wifi][Shops[i]].append(rank)
                else:
                    WifiAvgRankShop[wifi][Shops[i]]= []
                    WifiAvgRankShop[wifi][Shops[i]].append(rank)
            else:
                WifiAvgRankShop[wifi] = {}
                WifiAvgRankShop[wifi][Shops[i]] = []
                WifiAvgRankShop[wifi][Shops[i]].append(rank)

            if arr[j][0] in ShopDicWifi[shop_id]:
                ShopDicWifi[shop_id][arr[j][0]] = ShopDicWifi[shop_id][arr[j][0]] + 1
            else:
                ShopDicWifi[shop_id][arr[j][0]] = 1

    for wifi in WifiAvgSignalShopFirst:
        for shop in WifiAvgSignalShopFirst[wifi]:
            sum = 0.0
            arr = WifiAvgSignalShopFirst[wifi][shop]
            for j in range(0, len(arr)):
                sum = sum + arr[j]
            WifiAvgSignalShopFirst[wifi][shop] = sum / len(arr)
    for wifi in WifiAvgSignalShopSecond:
        for shop in WifiAvgSignalShopSecond[wifi]:
            sum = 0.0
            arr = WifiAvgSignalShopSecond[wifi][shop]
            for j in range(0, len(arr)):
                sum = sum + arr[j]
            WifiAvgSignalShopSecond[wifi][shop] = sum / len(arr)

    for wifi in WifiAvgRankShop:
        for shop in WifiAvgRankShop[wifi]:
            sum = 0.0
            arr = WifiAvgRankShop[wifi][shop]
            for j in range(0,len(arr)):
                sum = sum + arr[j]
            WifiAvgRankShop[wifi][shop] = sum/len(arr)

    idx = 0
    WifiDic = {}
    RevertWifiDic = {}

    for shop in ShopDicWifi:
            wifis = ShopDicWifi[shop]
            for wifi in wifis:
                if wifi not in WifiDic:
                    WifiDic[wifi] = idx
                    RevertWifiDic[idx] = wifi
                    idx = idx + 1


    for i in range(0,Len):
        arr = []
        cur = result[i]
        shop_id =Shops[i]

        for wifi in cur:
            arr.append([wifi,cur[wifi]])

        arr.sort(lambda x, y: cmp(x[1], y[1]),reverse=True)
        #arr = arr[0:10]
        for i in range(0,len(arr)):
            wifi = arr[i][0]
            if wifi in WifiDicShop:
                if shop_id in WifiDicShop[wifi]:
                    WifiDicShop[wifi][shop_id] = WifiDicShop[wifi][shop_id] + 1
                else:
                    WifiDicShop[wifi][shop_id] = 1
            else:
                WifiDicShop[wifi] = {}
                WifiDicShop[wifi][shop_id] = 1

    '''
    for wifi in WifiDicShop:
        shopsets = WifiDicShop[wifi]
        arr = []
        for shop in shopsets:
            arr.append([shop,shopsets[shop]])
        arr.sort(lambda x,y:cmp(x[1],y[1]),reverse=True)
        arr = arr[0:7]
        NewShopSets = {}
        for i in range(0,len(arr)):
            NewShopSets[arr[i][0]] = arr[i][1]
        WifiDicShop[wifi] = NewShopSets
    '''
    return ShopDicWifi,WifiDicShop,ShopCnt,WifiDic,RevertWifiDic,WifiConnect,WifiConnectShop,\
           WifiAvgSignalShop,WifiMedianSignalShop,WifiConnectSingalShop,WifiAvgRankShop,WifiDicShopFirst,WifiDicShopSecond,WifiAvgSignalShopFirst,WifiAvgSignalShopSecond



def ExtendTrain(train,wifidic):
    values = train['wifi_infos'].values
    Times = train['time_stamp'].values
    Len = len(values)
    result = []
    for i in  range(0,Len):
        tokens = values[i].split(";")
        dic = {}
        arr = []
        for x in tokens:
            xr = x.split("|")
            if (xr[0] in wifidic):
                arr.append([xr[0],float(xr[1])])
            if (xr[2] == 'true'):
                dic['ConnectInfos'] = xr[0]
                dic['Connect'] = int(wifidic[xr[0]])
                dic['ConnectNumber'] = float(WifiConnect[xr[0]])

        if 'Connect' not in dic:
            dic['Connect'] = np.nan
            dic['ConnectInfos'] = np.nan
        dic['time_stamp'] = Times[i]

        dic = AddTimeFeature(dic)

        arr.sort(lambda x,y:cmp(x[1],y[1]),reverse=True)
        #arr = arr[0:10]
        for j in range(0,len(arr)):
            t = arr[j]
            dic[t[0]] = t[1]

        for j in range(0,len(arr)):
            if arr[j][0] in wifidic:
                dic['wifi_infos'+str(j)] = int(wifidic[arr[j][0]])
        result.append(dic)

    L = train['longitude'].values
    R = train['latitude'].values
    shop_id = train['shop_id'].values

    for i in range(0,Len):
        dic = result[i]
        dic['longitude'] = L[i]
        dic['latitude'] = R[i]
        dic['shop_id'] = shop_id[i]
        result[i] = dic
    return result


def ExtendTest(test,wifidic,FinalFeature):
    values = test['wifi_infos'].values
    L = test['longitude'].values
    R = test['latitude'].values
    RowId = test['row_id'].values
    Times = test['time_stamp'].values
    Len = len(values)
    result = []
    for i in  range(0,Len):
        tokens = values[i].split(";")
        dic = {}
        dic['longitude'] = L[i]
        dic['latitude'] = R[i]
        dic['row_id'] = RowId[i]
        dic['time_stamp'] = Times[i]

        dic = AddTimeFeature(dic)

        arr = []
        for x in tokens:
            xr = x.split("|")
            if (xr[0] in wifidic):
                arr.append([xr[0],float(xr[1])])
            if (xr[2] == 'true'):
                if xr[0] in wifidic:
                    dic['Connect'] = int(wifidic[xr[0]])
                    dic['ConnectInfos'] = xr[0]
                    if xr[0] in WifiConnect:
                        dic['ConnectNumber'] = float(WifiConnect[xr[0]])

            if 'Connect' not in dic:
                dic['Connect'] = np.nan
                dic['ConnectInfos'] = np.nan

        arr.sort(lambda x,y:cmp(x[1],y[1]),reverse=True)

        #arr=arr[0:10]

        for j in range(0,len(arr)):
            t = arr[j]
            dic[t[0]] = t[1]

        for j in range(0,len(arr)):
            if arr[j][0] in wifidic:
                dic['wifi_infos'+str(j)] = int(wifidic[arr[j][0]])

        result.append(dic)
    return result

def stop():
    time.sleep(1200000)

def GetShopSets(dic):

    ShopSets = {}
    ShopResult = {}

    for wifi in dic:
        if wifi in WifiDicShop:
            shopsets = WifiDicShop[wifi]
            for shop in shopsets:
                if shop in ShopSets:
                    ShopSets[shop] = ShopSets[shop] + shopsets[shop]
                else:
                    ShopSets[shop] = shopsets[shop]

    arr1 = []
    for shop in ShopSets:
        arr1.append([shop,ShopSets[shop]])
    arr1.sort(lambda x,y:cmp(x[1],y[1]),reverse=True)
    arr1 = arr1[0:12]
    for i in range(0,len(arr1)):
        ShopResult[arr1[i][0]] = arr1[i][1]

    return ShopResult

def GetDistance(dic,ShopCurs):
    arr = []
    las = ShopCurs['latitude'].values
    lgs = ShopCurs['longitude'].values
    shops = ShopCurs['shop_id'].values
    ShopSets = {}
    for i in range(0,len(shops)):
        if(shops[i] in ShopCnt):
            dis = ComputeDistance(las[i],lgs[i],dic['latitude'],dic['longitude'])
            arr.append([shops[i],dis])
    arr.sort(lambda x,y:cmp(x[1],y[1]))
    arr = arr[0:40]
    for i in range(0,len(arr)):
        ShopSets[arr[i][0]] = True
    return ShopSets


def BuildTestSamples(Test):

    Result = []

    for i in range(0,len(Test)):
        ShopSets = GetShopSets(Test[i])

        if (len(ShopSets) == 0):
            ShopSets = GetDistance(dic,ShopCurs)
        for shop in ShopSets:
            dic = deepcopy(Test[i])
            shopcurs = ShopCurs[ShopCurs['shop_id']==shop]
            las = shopcurs['latitude'].values
            lgs = shopcurs['longitude'].values
            dic['shop_id'] = ShopDic[shop]
            shop_id = shop
            dic = AddShopInfos(dic, shop_id)
            dic = AddCntAndRitio(dic,shop_id)
            dic = AddWifiCnt(dic,shop_id)
            dic = AddConnectInofos(dic,shop_id)
            dic = AddAvgsingalDelta(dic, shop_id)
            Result.append(dic)

    return Result


def AddConnectInofos(dic,shop_id):
    cur = dic['ConnectInfos']

    if cur in WifiConnectShop and shop_id in WifiConnectShop[cur]:
        dic['ConnectShopNumber'] = WifiConnectShop[cur][shop_id]
    else:
        dic['ConnectShopNumber'] = 0

    return dic

def AddAvgsingalDelta(dic,shop_id):
    arr = []
    for wifi in dic:
        if wifi in WifiDic:
            arr.append([wifi,dic[wifi]])
    arr.sort(lambda x,y:cmp(x[1],y[1]),reverse=True)

    for i in range(0,len(arr)):
        if arr[i][0] in WifiAvgSignalShop and shop_id in WifiAvgSignalShop[arr[i][0]]:
            dic['WifiAvgDelta'+str(i)] = abs(dic[arr[i][0]] - WifiAvgSignalShop[arr[i][0]][shop_id])
            dic['WifiAvg'+str(i)] = WifiAvgSignalShop[arr[i][0]][shop_id]
        else:
            dic['WifiAvgDelta'+str(i)] = np.nan
            dic['WifiAvg'+str(i)] = np.nan

        if arr[i][0] in WifiMedianSignalShop and shop_id in WifiMedianSignalShop[arr[i][0]]:
            dic['WifiMedianDelta'+str(i)] = abs(dic[arr[i][0]] - WifiMedianSignalShop[arr[i][0]][shop_id])
            dic['WifiMedian'+str(i)] = WifiMedianSignalShop[arr[i][0]][shop_id]
        else:
            dic['WifiMedianDelta'+str(i)] = np.nan
            dic['WifiMedian'+str(i)] = np.nan
        if arr[i][0] in WifiAvgRankShop and shop_id in WifiAvgRankShop[arr[i][0]]:
            dic['WifiAvgRankDelta'+str(i)] = i - WifiAvgRankShop[arr[i][0]][shop_id]
            dic['WifiAvgRank'+str(i)] = WifiAvgRankShop[arr[i][0]][shop_id]
        else:
            dic['WifiAvgRankDelta'+str(i)] = np.nan
            dic['WifiAvgRank'+str(i)] = np.nan
    return dic


def AddPositive(inputdic):

    dic = deepcopy(inputdic)

    shopcur = ShopCurs[ShopCurs['shop_id'] == dic['shop_id']]
    las = shopcur['latitude'].values
    lgs = shopcur['longitude'].values
    dic['label'] = 1
    shop_id = dic['shop_id']
    dic = AddConnectInofos(dic,shop_id)
    dic = AddShopInfos(dic, shop_id)
    dic = AddCntAndRitio(dic,shop_id)
    dic = AddWifiCnt(dic,shop_id)
    dic = AddAvgsingalDelta(dic,shop_id)

    dic['shop_id'] = ShopDic[dic['shop_id']]
    return dic

def AddWifiCnt(dic,shop_id):

    arr = []
    for cur in dic:
        if cur in WifiDic:
            arr.append(cur)

    wifis = ShopDicWifi[shop_id]

    sum = 0.0
    for i in range(0,len(arr)):
        if arr[i] in wifis:
            sum = sum + 1
    dic['WifiCnt'] = sum

    if (dic['Connect'] is np.nan):
        dic['ConnectNum'] = np.nan
        dic['ConnectNumRitio'] = np.nan
    else:
        wifi = RevertWifiDic[dic['Connect']]

        if wifi in WifiConnectSingalShop and shop_id in WifiConnectSingalShop[wifi]:
            dic['WifiConnectSigDelta11'] = abs(dic[wifi] - WifiConnectSingalShop[wifi][shop_id])
            dic['WifiConnectSig11'] = WifiConnectSingalShop[wifi][shop_id]
        if wifi in WifiAvgSignalShop and shop_id in WifiAvgSignalShop[wifi]:
            dic['WifiConnectSigDelta12'] = abs(dic[wifi] - WifiAvgSignalShop[wifi][shop_id])
            dic['WifiConnectSig12'] = WifiAvgSignalShop[wifi][shop_id]

        if wifi in wifis:
            Num = float(wifis[wifi])
        else:
            Num = 0.0
        dic['ConnectNum'] = Num
        dic['ConnectNumRitio'] = Num/ShopCnt[shop_id]

    return dic


def AddShopInfos(dic,shop_id):


    dic['ShopCnt'] = ShopCnt[shop_id]
    wifis = ShopDicWifi[shop_id]
    dic['ShopWifiCnt'] = float(len(wifis))
    return dic


def AddNegative(inputdic,ShopSets):
    Result = []
    for shop in ShopSets:
        dic = deepcopy(inputdic)
        shopcur = ShopCurs[ShopCurs['shop_id'] == shop]
        las = shopcur['latitude'].values
        lgs = shopcur['longitude'].values

        shop_id = shop
        dic = AddCntAndRitio(dic,shop_id)
        dic = AddShopInfos(dic,shop_id)
        dic = AddWifiCnt(dic,shop_id)
        dic = AddConnectInofos(dic,shop_id)
        dic = AddAvgsingalDelta(dic,shop_id)
        if (dic['shop_id'] == shop):
            continue
        else:
            dic['label'] = 0
            dic['shop_id'] = ShopDic[shop]
            Result.append(dic)
    #Result = shuffle(Result)
    #Result = Result[0:12]

    return Result

def AddCntAndRitio(dic,shop_id):
    arr = []
    for wifi in dic:
        if wifi in WifiDicShop:
            arr.append([wifi,dic[wifi]])

    arr.sort(lambda x,y:cmp(x[1],y[1]),reverse=True)
    for i in range(0,len(arr)):
        wifi = arr[i][0]
        dic['sig' + str(i)] = arr[i][1]
        if wifi in WifiDicShop and shop_id in WifiDicShop[wifi]:
            dic['wifi_Cnt'+str(i)] =  WifiDicShop[wifi][shop_id]
            dic['wifi_Cnt_Ritio'+str(i)] = WifiDicShop[wifi][shop_id]/ShopCnt[shop_id]
        if wifi in WifiDicShopFirst and shop_id in WifiDicShopFirst[wifi]:
            dic['wifi_CntFirst'+str(i)] =  WifiDicShopFirst[wifi][shop_id]
            dic['wifi_Cnt_RitioFirst'+str(i)] = WifiDicShopFirst[wifi][shop_id]/ShopCnt[shop_id]
        if wifi in WifiDicShopSecond and shop_id in WifiDicShopSecond[wifi]:
            dic['wifi_CntSecond'+str(i)] = WifiDicShopSecond[wifi][shop_id]
            dic['wifi_Cnt_RitioSecond'+str(i)] = WifiDicShopSecond[wifi][shop_id]/ShopCnt[shop_id]
        if (wifi in WifiConnectShop and shop_id in WifiConnectShop[wifi]):
            dic['WifiConnectNumber'+str(i)] = WifiConnectShop[wifi][shop_id]
        if (wifi in WifiConnectSingalShop and shop_id in WifiConnectSingalShop[wifi]):
            dic['WifiConnectSigDelta'+str(i)] = abs(arr[i][1] - WifiConnectSingalShop[wifi][shop_id])
            dic['WifiConnectSig'+str(i)] = WifiConnectSingalShop[wifi][shop_id]
        if (wifi in WifiAvgSignalShopFirst and shop_id in WifiAvgSignalShopFirst[wifi]):
            dic['WifiAvgSigDeltaFirst'+str(i)] = abs(arr[i][1] - WifiAvgSignalShopFirst[wifi][shop_id])
            dic['WifiAvgSigFirst'+str(i)] = WifiAvgSignalShopFirst[wifi][shop_id]
        if (wifi in WifiAvgSignalShopSecond and shop_id in WifiAvgSignalShopSecond[wifi]):
            dic['WifiAvgSigDeltaSecond'+str(i)] = abs(arr[i][1] - WifiAvgSignalShopSecond[wifi][shop_id])
            dic['WifiAvgSigSecond'+str(i)] = WifiAvgSignalShopSecond[wifi][shop_id]


    return dic

def BuildTrainSamples(Positive):

    Result = []
    for  i in range(0,len(Positive)):
        ShopSets = GetShopSets(Positive[i])

        dicpositive = AddPositive(Positive[i])
        Result.append(dicpositive)

        dicnegatives = AddNegative(Positive[i],ShopSets)
        Result.extend(dicnegatives)

    return Result

def BuildWifi(TrainDic):
    dic = {}
    idx = 0
    for t in TrainDic:
        dic[t] = idx
        idx =idx +1
    return dic


def time_map(time):
    time = time.hour
    time = int(time)
    if (time >= 23) & (time <=6):
        return 0
    elif (time >=7) & (time<=10):
        return 1
    elif (time >=11) & (time <=14):
        return 2
    elif (time >=15) & (time <= 18):
        return 3
    elif (time >=19) & (time <=21):
        return 4
    else:
        return 5

def AddTimeFeature(dic):
    curtime = dic['time_stamp']
    token = datetime.strptime(curtime,"%Y-%m-%d %H:%M")
    dic['week'] = int(token.weekday())
    dic['hourtime'] = time_map(token.time())
    return dic


def StartTrain(train,feature):
    print("here")
    train = pd.DataFrame(train)

    print("here")

    train_y = train['label']

    train_x = train[feature]
    #train_x = train_x.fillna(-999)

    models = []
    fold = 1

    for i in range(fold):
        params = {
            'eta': 0.1,  # use 0.002
            'max_depth': 9,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': i,
            'silent': True
        }
        x1, x2, y1, y2 = model_selection.train_test_split(train_x, train_y, test_size=0.3, random_state=i)
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'test')]
        model = xgb.train(params, xgb.DMatrix(x1, y1), 1000, watchlist, maximize=False, verbose_eval=50,
                      early_stopping_rounds=50)
        models.append(model)
        features = list(train_x.columns)
        create_feature_map(features)
        importance = model.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        df.to_csv("feature_importance/" + str(name) + "feat_importance.csv", index=False)
        #model = xgb.train(params, xgb.DMatrix(x1, y1), 50, watchlist)

    return models


def WriteCsv(path,Res):
    fout  = open(path,"w+")
    coutf = csv.writer(fout)
    coutf.writerow(['row_id','shop_id'])

    for idx in Res:
        coutf.writerow(idx)

    return

Mall_Dic = {}

train = pd.read_csv(train_path)

shop_info = pd.read_csv(shop_path)

testIn = pd.read_csv(test_path)

testIn[['row_id']] = testIn[['row_id']].astype(str)

train = train[train['mall_id']=='m_6587']

#0 user_id 1 longitude 2 latitude 3
def GetWifiDic(input):
    dic = {}
    values = input['wifi_infos'].values

    for t in values:
        tokens = t.split(";")
        for x in tokens:
            xr = x.split("|")
            wifi = xr[0]
            if (wifi in dic):
                dic[wifi] = dic[wifi] + 1
            else:
                dic[wifi] = 1
    AllWifi = {}
    DeleteWifi = {}
    for d in dic:
        if (dic[d]>10):
            AllWifi[d] = True
    return AllWifi

def FinalDic(traindic,testdic):
    dicres = {}
    for t in traindic:
        if t in testdic:
            dicres[t] = True
    return dicres



def BuildShopIds(train):
    shops = train['shop_id'].values
    dic = {}
    revertdic = {}
    idx = 0
    for i in range(0,len(shops)):
        shop  = shops[i]
        dic[shop] = idx
        revertdic[idx] = shop
        idx = idx + 1
    return dic,revertdic

def StartPredict(FinalTest,models,FinalFeature):
    InputTest = pd.DataFrame(FinalTest)

    Input = InputTest[FinalFeature]
    for i in range(0,len(models)):
        model = models[i]
        if (i == 0):
            pred =  model.predict(xgb.DMatrix(Input),model.best_ntree_limit)
        else:
            pred = pred + model.predict(xgb.DMatrix(Input),model.best_ntree_limit)
    pred /= len(models)

    InputTest['label'] = pred
    InputTest = InputTest.sort_values('label',ascending=False).drop_duplicates('row_id')
    Result = InputTest[['row_id','shop_id']]

    return Result

def RocoveryShopId(Result,RevertDic):

    arr = Result['row_id'].values
    shops = Result['shop_id'].values

    finalresult = []
    for i in range(0,len(Result)):
        dic = {}
        dic['row_id'] = arr[i]
        dic['shop_id'] = RevertDic[shops[i]]
        finalresult.append(dic)
    return finalresult


def RemoveWifi(TrainGroup):
    InputFeature = []
    for i in range(0,len(FinalFeature)):
        InputFeature.append(FinalFeature[i])
    InputFeature.append('label')

    Result = []

    for dic in TrainGroup:
        curdic = {}
        for cur in dic:
            if cur in InputFeature:
                curdic[cur] = dic[cur]
        Result.append(curdic)

    return Result


def GetFeatures():

    WifiInfos = []
    SortWifiRank = []
    SortWifiRankDelta = []
    SortWifiAvg = []
    SortWifiAvgDelta = []
    SortWifiCnts = []
    SortWifiCntRotio = []
    SortWifiMedianDelta = []
    SortWifiMedian = []
    SortWifiConnectNumber = []
    SortWifiConnectSig = []
    SortWifiConnectSigDelta = []
    SortWifiShopRankCnt = []
    SortWifiShopSecond = []
    SortWifiShopFirst = []
    SortWifiShopSecondRitio = []
    SortWifiShopFirstRitio = []

    SortWifiSignalShopFirst = []
    SortWifiSignalShopSecond = []
    SortWifiSignalShopFirstDelta = []
    SortWifiSignalShopSecondDelta = []


    for i in range(0,10):
        SortWifiCnts.append('wifi_Cnt'+str(i))
        SortWifiCntRotio.append('wifi_Cnt_Ritio'+str(i))
        SortWifiShopSecond.append('wifi_CntSecond'+str(i))
        SortWifiShopFirst.append('wifi_CntFirst'+str(i))
        SortWifiShopSecondRitio.append('wifi_Cnt_RitioSecond'+str(i))
        SortWifiShopFirstRitio.append('wifi_Cnt_RitioFirst' + str(i))
        #SortWifiAvg.append('WifiAvg'+str(i))
    for i in range(0,10):
        SortWifiAvgDelta.append('WifiAvgDelta'+str(i))
        SortWifiMedianDelta.append('WifiMedianDelta'+str(i))
        SortWifiConnectNumber.append('WifiConnectNumber'+str(i))
        SortWifiConnectSig.append('WifiConnectSig'+str(i))
        SortWifiConnectSigDelta.append('WifiConnectSigDelta'+str(i))
        SortWifiRankDelta.append('WifiAvgRankDelta'+str(i))
        SortWifiSignalShopFirstDelta.append('WifiAvgSigDeltaFirst'+str(i))
        SortWifiSignalShopSecondDelta.append('WifiAvgSigDeltaSecond' + str(i))
        SortWifiSignalShopFirst.append('WifiAvgSigFirst'+str(i))
        SortWifiSignalShopSecond.append('WifiAvgSigSecond' + str(i))

    for i in range(0,10):
        SortWifiAvg.append('WifiAvg'+str(i))
        SortWifiMedian.append('WifiMedian'+str(i))


    ConnectFeature = ['Connect','ConnectNum','ConnectNumRitio','ConnectNumber','ConnectShopNumber'
        ,'WifiConnectSigDelta11','WifiConnectSigDelta12','WifiConnectSig11','WifiConnectSig12']

    for i in range(0,10):
        WifiInfos.append('wifi_infos'+str(i))

    feature = ['longitude','latitude','shop_id','ShopCnt','ShopWifiCnt','WifiCnt']

    feature.extend(SortWifiShopFirstRitio)

    feature.extend(SortWifiShopRankCnt)


    feature.extend(SortWifiRankDelta)

    feature.extend(SortWifiAvg)

    feature.extend(SortWifiConnectNumber)

    feature.extend(SortWifiConnectSig)

    feature.extend(SortWifiConnectSigDelta)

    feature.extend(SortWifiMedian)

    feature.extend(SortWifiAvgDelta)

    feature.extend(SortWifiMedianDelta)

    feature.extend(ConnectFeature)

    feature.extend(SortWifiCnts)

    feature.extend(SortWifiCntRotio)

    return feature

ResulT = pd.DataFrame(columns=['row_id','shop_id'])

AllTotalTestRight = 0
AllTotalLen = 0
AllTotalAvg_Shop_Num = 0

for name ,mall in train.groupby("mall_id"):
    print(name)
    test = testIn[testIn['mall_id'] == name]
    ShopCurs = shop_info[shop_info['mall_id'] == name]
    TOTALNUMBER = len(mall)
    Mall_Dic[name] = True
    ShopModel = {}
    features_x = []
    ShopName = {}
    ShopWifi ={}
    Train_x = {}
    ShopFeature = {}
    DeleteWifiArray = {}
    TestWifi = []
    TrainWifi = []

    pd.set_option('display.max_columns', None)

    pd.set_option('display.max_rows', None)

    mallvalue = mall.values

    ShopDic,RevertShopDic = BuildShopIds(ShopCurs)

    print("BUILD shop id end !")

    print(len(ShopDic))

    ShopDicWifi,WifiDicShop,ShopCnt,WifiDic,RevertWifiDic,WifiConnect\
        ,WifiConnectShop,WifiAvgSignalShop,WifiMedianSignalShop,WifiConnectSingalShop,\
    WifiAvgRankShop,WifiDicShopFirst,WifiDicShopSecond,WifiAvgSignalShopFirst,WifiAvgSignalShopSecond = BuildShopDicWifi(mall)

    print("Build Candicate End!")

    print(len(WifiDic))
    print(len(WifiDicShop))

    for name1,shop in mall.groupby("shop_id"):
        ShopName[name1] = True
        CurShop = ExtendTrain(shop,WifiDic)
        Train_x[name1] = CurShop

    print("Extend Train End!")
    TrainGroup = []

    for name2 in Train_x:
        ShopSamples = BuildTrainSamples(Train_x[name2])
        TrainGroup.extend(ShopSamples)

    print("Build Train End!")
    FinalFeature = GetFeatures()

    TrainGroup = RemoveWifi(TrainGroup)

    print("Remove Wifi End!")

    models  = StartTrain(TrainGroup,FinalFeature)

    print(FinalFeature)

    print("Train End!")

    #Test = ExtendTest(test,WifiDic)

    Test = ExtendTest(test,WifiDic,FinalFeature)

    print("Extend Test End!")
    #r = BuildTestSamples(Test)
    FinalTest = BuildTestSamples(Test)

    print("Build Test End!")

    Result_x = StartPredict(FinalTest,models,FinalFeature)

    Result_x = RocoveryShopId(Result_x,RevertShopDic)

    Result_x = pd.DataFrame(Result_x)

    print(Result_x.head())

    ResulT = ResulT.append(Result_x)

    Result_x.to_csv("result/predict" + str(name) + str(".csv"), index=False)

ResulT.to_csv("result/result.csv", index=False)

ResulT.to_csv("../../data/Split/result.csv", index=False)
