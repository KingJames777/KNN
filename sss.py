import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from urllib import request
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 下载数据
def downloadData():
    if (not(os.path.exists('wine.data'))):
        print('Downloading with urllib\n')
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        request.urlretrieve(url,'./wine.data')
    else:
        print('Wine.data exists!\n')

# 数据预处理
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr =open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine[1:]))  #将字符list转换为float list
        dataMat.append(fltLine)
        labelLine = int(curLine[0])
        labelMat.append(labelLine)
    return np.array(dataMat),np.array(labelMat) #返回numpy数组

# 数据可视化（进行LDA降维）
def visualData(dataMat,labelMat):
    dataMat_norm = preprocessing.normalize(dataMat,norm='l2')   #数据归一化处理
    lda = LinearDiscriminantAnalysis(n_components=2)
    dataMat_new = lda.fit_transform(dataMat_norm,labelMat)
    plt.scatter(dataMat_new[:,0],dataMat_new[:,1],c=labelMat)
    plt.show()
    return dataMat_new

def main():
    downloadData()  # 下载数据
    dataMat,labelMat = loadDataSet('wine.data')
    dataMat = visualData(dataMat,labelMat)
    # 交叉验证数据集划分
    x_train,x_test,y_train,y_test = train_test_split(dataMat,labelMat,test_size=0.4,random_state=1)
    # SVM生成和训练
    clf=svm.SVC(kernel='linear',C=1,degree=3)
    clf.fit(x_train,y_train)
    # SVM结果预测
    result = clf.predict(x_test)
    # SVM正确率验证
    print('Accuracy:',clf.score(x_test,y_test))

if __name__ == '__main__':
    main()
