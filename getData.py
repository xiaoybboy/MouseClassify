# -*- coding: utf-8 -*-
import math
import feature as ft
import pandas as pd

def WriteTrainDataToFile():
     #读取原始数据
    fr = open('dsjtzs_txfz_training.txt','r') #训练集路径
    arrayOfLines = fr.readlines() #训练集上的所有行
    ids = [];#存放数据序号
    dataList = []; #存放数据
    labelList = [];#存放标签
    
    for line in arrayOfLines:#每一行的数据
        id = line.split()[0]
        xyt_line = line.split()[1].split(";")#鼠标轨迹点
        targetPoint = line.split()[2]
        label = line.split()[3]
        
        data = []
        for xyt in xyt_line[:-1]:
            x = float(xyt.split(",")[0])
            y = float(xyt.split(",")[1])
            t = float(xyt.split(",")[2])
            data.append([x,y,t])
            
        ids.append(id)
        dataList.append(data)
        labelList.append(label)
     
    #提取鼠标速度和距离等特征数据
    SpeedMoveData = []#鼠标移动速度和距离数据
    for i in range(len(dataList)):
        lineData = dataList[i]
        
        if len(lineData) == 1:
            del(ids[i])
            del(labelList[i])
            continue
        
        linespeedmove = []#一个样本的数据
        for j in range(len(lineData)-1):
            xmove = abs(lineData[j+1][0] - lineData[j][0])#x方向的移动距离
            ymove = abs(lineData[j+1][1] - lineData[j][1])#y方向的移动距离
            smove = math.sqrt(xmove**2 + ymove**2)#和移动距离
            timemove = lineData[j+1][2] - lineData[j][2]
            if timemove <= 0:
                continue
            
            speed_x = abs(xmove/timemove)#x方向速度
            speed_y = abs(ymove/timemove)#y方向速度
            speed_s = math.sqrt(speed_x**2+speed_y**2) #和速度
            linespeedmove.append([speed_x,speed_y,speed_s,timemove,xmove,ymove,smove])
            
        SpeedMoveData.append(linespeedmove)#所有样本的数据
        
    #提取特征
    feature = [] #[data1,data2...]
    for speedMoveLine in SpeedMoveData:#每一行（每一个样本的数据）
        speed_x_list = []
        speed_y_list = []
        speed_s_list = []
        times = []
        move_x_list = []
        move_y_list = []
        move_s_list = []
        for i in range(len(speedMoveLine)):
            speedmovedata = speedMoveLine[i]
            speed_x = speedmovedata[0]
            speed_y = speedmovedata[1]
            speed_s = speedmovedata[2]
            time = speedmovedata[3]
            move_x = speedmovedata[4]
            move_y = speedmovedata[5]
            move_s = speedmovedata[6]
        
            speed_x_list.append(speed_x)
            speed_y_list.append(speed_y)
            speed_s_list.append(speed_s)
            times.append(time)
            move_x_list.append(move_x)
            move_y_list.append(move_y)
            move_s_list.append(move_s) 
        
        feature.append(ft.Get_FeatureList1(speed_x_list) + 
                   ft.Get_FeatureList1(speed_y_list) + 
                   ft.Get_FeatureList1(speed_s_list) +
                   ft.Get_FeatureList1(times) +
                   ft.Get_FeatureList1(move_x_list) +
                   ft.Get_FeatureList1(move_y_list) +
                   ft.Get_FeatureList1(move_s_list))
    
    df = pd.DataFrame(feature)
    df.astype(float)
    df.to_csv('feature/feature1.csv',header=False,index=False)
    
    labelListdf = pd.DataFrame(labelList)
    labelListdf.astype(float)
    labelListdf.to_csv('feature/label1.csv',header=False,index=False)

def WritePredictDataTofile():
    fr = open('dsjtzs_txfz_test1.txt','r') #预测集路径
    arrayOfLines = fr.readlines() #预测集上的所有行
    ids = [];#存放数据序号
    dataList = []; #存放数据
    
    for line in arrayOfLines:#每一行的数据
        id = line.split()[0]
        xyt_line = line.split()[1].split(";")#鼠标轨迹点
        targetPoint = line.split()[2]
        
        data = []
        for xyt in xyt_line[:-1]:
            x = float(xyt.split(",")[0])
            y = float(xyt.split(",")[1])
            t = float(xyt.split(",")[2])
            data.append([x,y,t])
            
        ids.append(id)
        dataList.append(data)
        
    SpeedMoveData = []#鼠标移动速度和距离数据
    
    for i in range(len(dataList)):
        LineData = dataList[i]#一个预测样本的数据（一行）
        
        if len(LineData) == 1:
            del(ids[i])
            continue

        linespeedmove = []#一个样本的数据
        for j in range(len(LineData)-1):
            xmove = abs(LineData[j+1][0] - LineData[j][0])
            ymove = abs(LineData[j+1][1] - LineData[j][1])
            smove = math.sqrt(xmove**2 + ymove**2)#和移动距离
            timeMove  = LineData[j+1][2] - LineData[j][2]
            if timeMove<=0:
                continue
            
            speed_x = abs(xmove/timeMove)#x方向速度
            speed_y = abs(ymove/timeMove)#y方向速度
            speed_s = math.sqrt(speed_x**2+speed_y**2) #和速度
            linespeedmove.append([speed_x,speed_y,speed_s,timeMove,xmove,ymove,smove])
        
        SpeedMoveData.append(linespeedmove)
        
        #提取特征
    predictfeature = [] #[data1,data2...]
    for prespeedMoveLine in SpeedMoveData:#每一行（每一个样本的数据）
        pspeed_x_list = []
        pspeed_y_list = []
        pspeed_s_list = []
        ptimes = []
        pmove_x_list = []
        pmove_y_list = []
        pmove_s_list = []
        for j in range(len(prespeedMoveLine)):
            pspeedmovedata = prespeedMoveLine[j]
            pspeed_x = pspeedmovedata[0]
            pspeed_y = pspeedmovedata[1]
            pspeed_s = pspeedmovedata[2]
            ptime = pspeedmovedata[3]
            pmove_x = pspeedmovedata[4]
            pmove_y = pspeedmovedata[5]
            pmove_s = pspeedmovedata[6]
        
            pspeed_x_list.append(pspeed_x)
            pspeed_y_list.append(pspeed_y)
            pspeed_s_list.append(pspeed_s)
            ptimes.append(ptime)
            pmove_x_list.append(pmove_x)
            pmove_y_list.append(pmove_y)
            pmove_s_list.append(pmove_s) 
        
        predictfeature.append(ft.Get_FeatureList1(pspeed_x_list) + 
                       ft.Get_FeatureList1(pspeed_y_list) + 
                       ft.Get_FeatureList1(pspeed_s_list) +
                       ft.Get_FeatureList1(ptimes) +
                       ft.Get_FeatureList1(pmove_x_list) +
                       ft.Get_FeatureList1(pmove_y_list) +
                       ft.Get_FeatureList1(pmove_s_list))
    
    df = pd.DataFrame(predictfeature)
    df.astype(float)
    df.to_csv('feature/predictfeature1.csv',header=False,index=False)
    
    idsdf = pd.DataFrame(ids)
    idsdf.to_csv('feature/predictids.csv',header=False,index=False)