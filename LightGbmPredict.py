# -*- coding: utf-8 -*-
import getData as gd
import feature as ft
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 

#获取训练集数据
print ("Read the Training data!")
ids,dataList,labelList = gd.getTrainDataFromFile("dsjtzs_txfz_training.txt")
SpeedMoveData = gd.getTrainSpeedMoveData(ids,dataList,labelList)

print ("get the features!")
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
        timemove = speedmovedata[3]
        move_x = speedmovedata[4]
        move_y = speedmovedata[5]
        move_s = speedmovedata[6]
        
        speed_x_list.append(speed_x)
        speed_y_list.append(speed_y)
        speed_s_list.append(speed_s)
        times.append(timemove)
        move_x_list.append(move_x)
        move_y_list.append(move_y)
        move_s_list.append(move_s) 
        
    feature.append(ft.Get_FeatureList(speed_x_list) + 
                   ft.Get_FeatureList(speed_y_list) + 
                   ft.Get_FeatureList(speed_s_list) +
                   ft.Get_FeatureList(times) +
                   ft.Get_FeatureList(move_x_list) +
                   ft.Get_FeatureList(move_y_list) +
                   ft.Get_FeatureList(move_s_list))

print ("数据预处理")

#把训练集数据转float
feature = np.array(feature)
feature = feature.astype(float)

labelList = np.array(labelList)
labelList = labelList.astype(int)

#预处理，进行归一化和PCA降维
#scaler = preprocessing.StandardScaler().fit(feature)
#train_data = preprocessing.minmax_scale(feature, feature_range=(0, 1))

#pca=PCA(n_components=20,whiten = True)#pca降到20维
#train_data=pca.fit_transform(feature)

train_X,test_X, train_y, test_y = train_test_split(feature, labelList, test_size = 0.2, random_state =33)

print('training start')
num_clfs = 5
clfs = []
for i in range(num_clfs):
    clf = BaggingClassifier(
            lgb.sklearn.LGBMClassifier(boosting_type='gbdt',
                                       objective = 'binary',
                                       max_bin = 255,
                                       n_estimators=2000,
                                       learning_rate=0.05,
                                       seed =i,
                                       silent=False,
                                       num_leaves=60 + 10 * i),
                               n_estimators=5, random_state=i, n_jobs=1, max_samples=0.6 + 0.01 * i)
    clf.fit(train_X, train_y)
    clfs.append(clf)
print('training  done')

#验证结果
trueMachineNum = 0
predictMachineNum = 0
predictMachineTrueNum=0
for sample in test_y:
    if sample == 0:
        trueMachineNum+=1
        
for i in range(len(test_X)):
    data = [test_X[i]]
    label = test_y[i]
    ans = [clf.predict(data) for clf in clfs]
    if ft.getFinalResult(ans) == 0:
        predictMachineNum+=1
        if label==0:
            predictMachineTrueNum+=1

P = predictMachineTrueNum/predictMachineNum
R = predictMachineTrueNum/trueMachineNum
F = 5*P*R/(2*P+3*R)*100
print (F)
'''
#读取预测数据
print('读取预测数据')
predictids,predictSpeedData=gd.getPredictDataFromFile("dsjtzs_txfz_test1.txt")
predictSpeedDataList = gd.getPredictSpeedMoveData(predictids,predictSpeedData)

print ("Read Done!")

#提取特征
print ("开始提取预测数据特征")
predictfeature = [] #[data1,data2...]
for predictspeedMoveLine in predictSpeedDataList:#每一行（每一个样本的数据）
    pspeed_x_list = []
    pspeed_y_list = []
    pspeed_s_list = []
    ptimes = []
    pmove_x_list = []
    pmove_y_list = []
    pmove_s_list = []
    for i in range(len(predictspeedMoveLine)):
        pspeedmovedata = predictspeedMoveLine[i]
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
    predictfeature.append(ft.Get_FeatureList(pspeed_x_list) + 
                   ft.Get_FeatureList(pspeed_y_list) + 
                   ft.Get_FeatureList(pspeed_s_list) +
                   ft.Get_FeatureList(ptimes)+ 
                   ft.Get_FeatureList(pmove_x_list) +
                   ft.Get_FeatureList(pmove_y_list) +
                   ft.Get_FeatureList(pmove_s_list))

#把训练集数据转float
predictfeature = np.array(predictfeature)
predict_data = predictfeature.astype(float)

#预处理，进行归一化和PCA降维
#scaler.transform(predictfeature)
#train_data = preprocessing.minmax_scale(feature, feature_range=(0, 1))
#predict_data=pca.fit_transform(predictfeature)

# 执行预测
print('predicting start')
#结果写到文件里面
fout = open("dsjtzs_txfzjh_preliminary.txt",'w')

print ("Start writing...")
for j in range(len(predict_data)):
    predata = [predict_data[j]]
    ans = [clf.predict(predata) for clf in clfs]
    if ft.getFinalResult(ans) == 0:
        fout.write(predictids[j]+'\n')
						
fout.close()
print ("All Done")
'''
