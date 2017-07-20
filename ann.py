# -*- coding: utf-8 -*-
import getDataFromFile as gdf
import feature as ft
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD
from sklearn.decomposition import PCA

#获取数据
ids,dataList,labelLists = gdf.getTrainDataFromTxt("dsjtzs_txfz_training.txt")
speedData = gdf.getTrainSpeed(ids,dataList,labelLists)

feature = [] #[data1,data2...]
for speedList in speedData:
    speed_x_list = [];speed_y_list = [];speed_s_list=[];times=[];
    for speed in speedList:
        speed_x = speed[0]
        speed_y = speed[1]
        speed_s = speed[2]
        speed_x_list.append(speed_x)
        speed_y_list.append(speed_y)
        speed_s_list.append(speed_s)
        times.append(speed[3])
    feature.append(ft.Get_FeatureList(speed_x_list) + ft.Get_FeatureList(speed_y_list) + ft.Get_FeatureList(speed_s_list)+ft.Get_FeatureList(times))

#数据归一化
train_data = preprocessing.minmax_scale(feature, feature_range=(0, 1))

pca=PCA(n_components=10,whiten=True)
train_data=pca.fit_transform(train_data)

train_X,test_X, train_y, test_y = train_test_split(train_data, labelLists, test_size = 0.1, random_state =33)

#建立模型
model = Sequential()
model.add(Dense(78, input_shape=(10,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
model.add(Activation('sigmoid'))

model.summary()

# 模型编译
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#ada = Adagrad(lr=0.01, epsilon=1e-06)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(np.array(train_X),np.array(train_y),epochs=20,batch_size=128,verbose=1)
score = model.evaluate(test_X, test_y, batch_size=128)
print (score)
'''
#获取预测数据
predictids,predictSpeedData=gdf.getRegressDataFromTxt("dsjtzs_txfz_test1.txt")
predictSpeedData = gdf.getRegressSpeedList(predictids,predictSpeedData)

#结果写到文件里面
fout = open("annpredict.txt",'w')

for i in range(len(predictSpeedData)):
    speedList = predictSpeedData[i]
    speed_x_list = [];speed_y_list = [];speed_s_list=[];times = [];
    predictfeature = [] #[data1,data2...]
    for speed in speedList:
        speed_x = speed[0]
        speed_y = speed[1]
        speed_s = speed[2]
        time = speed[3]
        speed_x_list.append(speed_x)
        speed_y_list.append(speed_y)
        speed_s_list.append(speed_s)
        times.append(time)
		
    predictfeature.extend(ft.Get_FeatureList(speed_x_list))
    predictfeature.extend(ft.Get_FeatureList(speed_y_list))
    predictfeature.extend(ft.Get_FeatureList(speed_s_list))
    predictfeature.extend(ft.Get_FeatureList(times))
	
    if model.predict(np.array(predictfeature)) == 0:
       fout.write(predictids[i]+'\n')
						
fout.close()
'''




