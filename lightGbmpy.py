# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier

#获取训练集数据
trainData = pd.read_csv('feature/feature1.csv',header=None)
label = pd.read_csv('feature/label1.csv',header=None)

print ("数据预处理")
train_X,test_X, train_y, test_y = train_test_split(trainData, label, test_size = 0.2,random_state = 0)

'''
print ('start training...')
param_test1 = {
   'num_leaves':range(2,10,1)
}
gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(  
        boosting_type='gbdt',        
        learning_rate =0.1,n_estimators=200,num_leaves =30,
        max_depth = 2,min_child_weight = 1,
        subsample=0.8,colsample_bytree=0.8,
        silent=0,reg_alpha = 1e-5,
        objective= 'binary', nthread=1,scale_pos_weight=1, 
        seed=0), 
param_grid = param_test1,     
scoring='roc_auc',n_jobs=1,cv=5)
gsearch1.fit(train_X,train_y)
print (gsearch1.best_params_)
'''

print('Start training...')
# train

gbm = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=3, 
                         max_depth=2, learning_rate=0.1, 
                         n_estimators=200, max_bin=255, 
                          objective='binary', 
                         subsample_freq=5,subsample=0.8,colsample_bytree=0.9, 
                         nthread=1,reg_alpha=1e-5, silent=0)

bagging_clf = BaggingClassifier(gbm, n_estimators=10, 
                                 max_samples=0.8, max_features=1.0,
                                 bootstrap=True, bootstrap_features=False,
                                 n_jobs=1)
bagging_clf.fit(train_X, train_y)

'''
#计算混淆矩阵
classtype = [0,1]
confusionMatrix = np.zeros((len(classtype),len(classtype)))

for i in range(len(test_X)):
    data = [test_X[i]]
    label = int(test_y[i])
    predict = int(bagging_clf.predict(np.array(data))[0])
    
    #预测分类    
    confusionMatrix[label][predict] += 1
    
print (confusionMatrix)
'''
#读取预测数据
predictData = pd.read_csv('feature/predictfeature1.csv',header=None)
predictids = pd.read_csv('feature/predictids.csv',header=None)
predictids = np.array(predictids)
# 执行预测
print('predicting start')
prediction = bagging_clf.predict(predictData)

#结果写到文件里面
fout = open("dsjtzs_txfzjh_preliminary.txt",'w')

print ("Start writing...")
for k in range(len(prediction)):
    if prediction[k] == 0:
        res = str(predictids[k][0])
        fout.write(res +'\n')
fout.close()
print ("All Done")