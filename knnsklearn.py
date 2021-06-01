# coding = utf8
from numpy import array
from sklearn import neighbors

import knn as KNN

def knn_sklearn_predict(newV, datasets, labels):
    # 调用机器学习库knn分类器算法
    knn = neighbors.KNeighborsClassifier()

    # 传入参数，特征数据，分类标签
    knn.fit(datasets, labels)
    # knn预测
    predictRes = knn.predict([newV])

    print('该访客认为成都天气是:', '非常热' if predictRes[0]==0 else '一般热')
    return predictRes

# 利用KNN分类器预测随机访客天气感知度
def Predict_temperature():
    # 调用模块下的方法，返回数据特征集和类标签
    datasets, labels = KNN.creat_datasets()
    # 2.采访新访客
    iceCream = float(input('Q:请问你今天吃了几个冰淇淋？\n'))
    drinkWater = float(input('Q:请问你今天喝了几瓶（杯）水？\n'))
    palyAct = float(input('Q:请问你今天户外活动几个小时？\n'))

    newV = array([iceCream, drinkWater, palyAct])
    knn_sklearn_predict(newV, datasets, labels)


if __name__=='__main__':
    Predict_temperature()