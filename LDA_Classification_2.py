import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import scipy.stats as ss ##高斯密度函数

bc=datasets.load_breast_cancer()
X=bc.data
y=bc.target

X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19541020,stratify=y)
##各类索引
index0=np.where(y_train==0)
index1=np.where(y_train==1)

##各类均值、协方差
u0=X_train[index0].mean(axis=0) 
u1=X_train[index1].mean(axis=0)
c0=np.cov(X_train[index0],rowvar=False)
c1=np.cov(X_train[index1],rowvar=False)

##类内散度矩阵和投影向量
sw=c0+c1
w=np.linalg.inv(sw).dot(u0-u1)

##降维后的数据及绘图
X_condensed=X_train.dot(w)
X_zero=np.zeros(X_condensed.shape)
plt.scatter(X_condensed[index0],X_zero[index0],c='navy',label='Negative',marker='x')
plt.scatter(X_condensed[index1],X_zero[index1],c='red',label='Positive',marker='+')
plt.xlim(X_condensed.min()-2,X_condensed.max()+2)
plt.grid()

##降维后的各类均值、方差
mu0=X_condensed[index0].mean()
mu1=X_condensed[index1].mean()
nu0=np.var(X_condensed[index0],ddof=1)
nu1=np.var(X_condensed[index1],ddof=1)

##绘制各类高斯曲线
x0=np.linspace(X_condensed.min()-2,X_condensed.max()+2,200)
x1=np.linspace(X_condensed.min()-2,X_condensed.max()+2,200)
y0=ss.norm.pdf(x0,mu0,nu0)
y1=ss.norm.pdf(x1,mu1,nu1)
plt.plot(x0,y0,'--',color='navy',label='Gaussian-negative')
plt.plot(x1,y1,'-.',color='red',label='Gaussian-positive')
plt.legend(loc='upper right')
plt.savefig('LDA for 2-Classification.png') ##这个要在plt.show()之前...
plt.show()

##比较验证集对应的各类高斯密度决定分类
X_test_condensed=X_test.dot(w)
y_pred=ss.norm.pdf(X_test_condensed,mu0,nu0)<ss.norm.pdf(X_test_condensed,mu1,nu1)
print(accuracy_score(y_test,y_pred))

##sklearn的正确率
lda = LinearDiscriminantAnalysis(n_components=1)
X_sklearn=lda.fit_transform(X_train,y_train)
print(lda.score(X_test,y_test))











