# coding = utf8

import numpy as np
from numpy import *

import math


# 创建数据源，返回数据集和类标签
def creat_dataset():
    datasets = array([[8, 4, 2], [7, 1, 1], [1, 4, 4], [3, 0, 5]])  #数据集
    labels = ['非常热', '非常热', '一般热', '一般热']  #类标签
    return datasets, labels


#构造KNN分类器
def knn_Classifier(newV, datasets,labels, k):
    import operator
    # 1.获取新的样本数据
    # 2.获取样本库的数据
    # 3.选择k至值
    # 4.计算样本数据与样本库数据之间的距离
    SqrtDist = EuclideanDistance3(newV, datasets)
    # 5.根据距离进行排序，按照列向量排序
    sortdDistIndexs = SqrtDist.argsort(axis=0)
    #print(sortdDistIndexs)
    # 6.针对k个点，统计各个类别的数量
    classCount = {} # 统计各个类别分别的数量
    for i in range(k):
        # 根据距离排序索引值找到类标签
        votelabel = labels[sortdDistIndexs[i]]
        #print(sortdDistIndexs[i], votelabel)
        # 统计类标签的键值对
        classCount[votelabel] = classCount.get(votelabel, 0)+1
    #print(classCount)
    # 7.投票机制，少数服从多数原则，输出类别
    # 对各个分类字典进行排序，降序，itemgetter按照value排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print(newV, 'KNN投票预测结果是:', sortedClassCount[0][0])
    return sortedClassCount[0][0]


# 欧式距离计算3
def EuclideanDistance3(newV, datasets):
    # 1.获取数据向量的行向量维度和列向量维度
    rowsize, colsize = datasets.shape
    # 2.各特征向量间作差值
    diffMat = tile(newV, (rowsize, 1)) - datasets
    #print(diffMat)
    # 3.对差值平方
    sqDiffMat = diffMat ** 2
    #print(sqDiffMat)
    # 4.差值平方和进行开方
    SqrtDist = sqDiffMat.sum(axis=1) ** 0.5
    #print(SqrtDist)
    return SqrtDist

# 利用KNN分类器预测随机访客天气感知度
def Predict_temperature():
    # 1.创建数据集和类标签
    datasets, labels = creat_dataset()
    # 2.采访新访客
    iceCream = float(input('Q:请问你今天吃了几个冰淇淋？\n'))
    drinkWater = float(input('Q:请问你今天喝了几瓶（杯）水？\n'))
    palyAct = float(input('Q:请问你今天户外活动几个小时？\n'))

    newV = array([iceCream, drinkWater, palyAct])
    res = knn_Classifier(newV, datasets, labels, 3)
    print("该访客认为成都天气是:", res)



if __name__=='__main__':
    # 5.利用KNN分类器预测随机访客天气感知度
    Predict_temperature()