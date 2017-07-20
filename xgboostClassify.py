# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier

#获取训练集数据
trainData = pd.read_csv('feature/feature1.csv')
label = pd.read_csv('feature/label1.csv')

print ("数据预处理")
train_X,test_X, train_y, test_y = train_test_split(trainData, label, test_size = 0.2,random_state = 0)
'''
param_test1 = {
  'learning_rate':[0.01, 0.003, 0.1, 0.2, 0.5],
}

gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(
                            n_estimators=200, silent=0,learning_rate=0.1,
                            objective='binary:logistic', 
                            booster='gbtree', n_jobs=1, 
                            nthread=None,gamma=0.1,
                            max_delta_step=0, min_child_weight=1,
                            subsample=0.7,colsample_bytree=0.9,
                            reg_alpha=0.1, reg_lambda=1, 
                            scale_pos_weight=1, base_score=0.5, 
                            random_state=0, seed=None, missing=None), 
                            param_grid = param_test1,     
                            scoring='roc_auc',n_jobs=1,cv=5)
gsearch1.fit(train_X,train_y)
print (gsearch1.best_params_)
'''
gbm = xgb.XGBClassifier(n_estimators=200, silent=0,learning_rate=0.1,
                        objective='binary:logistic', 
                        booster='gbtree', n_jobs=1, 
                        nthread=None,gamma=0.1,
                        max_delta_step=0, min_child_weight=1,
                        subsample=0.7,colsample_bytree=0.9,
                        reg_alpha=0.1, reg_lambda=1, 
                        scale_pos_weight=1, base_score=0.5, 
                        random_state=0, seed=None, missing=None)

bagging_clf = BaggingClassifier(gbm, n_estimators=10, 
                                 max_samples=0.8, max_features=1.0,
                                 bootstrap=True, bootstrap_features=False,
                                 n_jobs=1)
bagging_clf.fit(train_X, train_y)

#预测结果
preds = bagging_clf.predict(test_X)

#验证结果
trueMachineNum = 0
predictMachineNum = 0
predictMachineTrueNum=0
for sample in test_y:
    if sample == 0:
        trueMachineNum+=1
        
for i in range(len(preds)):
    if preds[i] < 0.5:
        predictMachineNum+=1
        if test_y[i]==0:
            predictMachineTrueNum+=1  

P = predictMachineTrueNum/predictMachineNum
R = predictMachineTrueNum/trueMachineNum
F = 5*P*R/(2*P+3*R)*100
print (F)

#读取预测数据
predictData = pd.read_csv('feature/predictfeature1.csv')
predictids = pd.read_csv('feature/predictids.csv')

# 执行预测
print('predicting start')
prediction = bagging_clf.predict(predictData)
#结果写到文件里面
fout = open("dsjtzs_txfzjh_preliminary.txt",'w')

print ("Start writing...")
for k in range(len(prediction)):
    if prediction[k] <0.5:
        fout.write(predictids[k]+'\n')
						
fout.close()
print ("All Done")
