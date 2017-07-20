# -*- coding: utf-8 -*-
'''
feature.py
'''
from sklearn.learning_curve import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#最大数
def Get_Max(list):
    return max(list)

#最小数
def Get_Min(list):
    return min(list)

#极差
def Get_Range(list):
    return max(list) - min(list)

#中位数
def get_median(data):
   data = sorted(data)
   size = len(data)
   if size % 2 == 0: # 判断列表长度为偶数
       median = (data[size//2]+data[size//2-1])/2
   if size % 2 == 1: # 判断列表长度为奇数
       median = data[(size-1)//2]
   return median

#众数(返回多个众数的平均值)
def Get_Most(list):
    most=[]
    item_num = dict((item, list.count(item)) for item in list)
    for k,v in item_num.items():
        if v == max(item_num.values()):
           most.append(k)
    return sum(most)/len(most)

#获取平均数
def Get_Average(list):
	sum = 0
	for item in list:
		sum += item
	return sum/len(list)

#获取方差
def Get_Variance(list):
	sum = 0
	average = Get_Average(list)
	for item in list:
		sum += (item - average)**2
	return sum/len(list)

#获取n阶原点距
def Get_NMoment(list,n):
    sum=0
    for item in list:
        sum += item**n
    return sum/len(list)

#返回速度特征列表
'''
def Get_FeatureList(list):
	return [Get_Min(list),Get_Max(list),Get_Most(list),Get_Average(list),Get_Variance(list),Get_Range(list),get_median(list)]
'''
def Get_FeatureList1(list):
	return [Get_Most(list),Get_Range(list),Get_Average(list),Get_Variance(list),get_median(list)]

if __name__ == "__main__":
    list = [1,2,3,4,5,6]
    median = get_median(list)
