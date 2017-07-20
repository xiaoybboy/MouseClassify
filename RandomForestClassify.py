# -*- coding: utf-8 -*-
import getData as gd
import feature as ft
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import StratifiedKFold


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
print ("数据预处理")

#把训练集数据转float
feature = np.array(feature)
feature = feature.astype(float)

labelList = np.array(labelList)
labelList = labelList.astype(int)

train_X,test_X, train_y, test_y = train_test_split(feature, labelList, test_size = 0.3,random_state = 0)

# 用随机森林选择特征
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train_X, train_y)

# 选取特征
model = SelectFromModel(clf, prefit=True)
train_X = model.transform(train_X)
print (train_X.shape)

'''画特征重要性的图
train_X = pd.DataFrame(train_X)

features = pd.DataFrame()
features['feature'] = train_X.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20, 20))
'''

#读取预测数据
print('读取预测数据')
predictids,predictSpeedData=gd.getPredictDataFromFile("dsjtzs_txfz_test1.txt")
predictSpeedDataList = gd.getPredictSpeedMoveData(predictids,predictSpeedData)

print ("get the features!")
#提取特征
predictfeature = [] #[data1,data2...]
for prespeedMoveLine in predictSpeedDataList:#每一行（每一个样本的数据）
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

#把训练集数据转float
predictfeature = np.array(predictfeature)
predict_data = predictfeature.astype(float)

predict_data = model.transform(predict_data)

# 训练模型
print('开始训练')
parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50,10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [3, 5, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
forest = RandomForestClassifier(verbose=10)
cross_validation = StratifiedKFold(train_y, n_folds=5)

grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)
grid_search.fit(train_X, train_y)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# 执行预测
print('predicting start')
#结果写到文件里面
fout = open("dsjtzs_txfzjh_preliminary.txt",'w')
num= 0

print ("Start writing...")
for k in range(len(predict_data)):
    predata = [predict_data[k]]
    if grid_search.predict(predata) == 0:
        num+=1
        fout.write(predictids[k]+'\n')
						
fout.close()
print ("All Done")










