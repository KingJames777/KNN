import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal ##多维高斯密度函数
from sklearn.metrics import accuracy_score

lw=load_wine()
X=lw.data
y = lw.target
X=(X-X.mean(axis=0))/X.std(axis=0)
X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901120,stratify=y)

##各类索引和各类协方差
index,cov=[],[]
for i in range(3):
      index=np.where(y_train==i)
      temp=np.cov(X_train[index],rowvar=False)
      temp*=len(index[0])  ##协方差不除以样本数
      cov.append(temp)

##类内协方差##整体协方差##类间协方差
sw=sum(cov)
st=(np.cov(X_train,rowvar=False))*len(X_train)
sb=st-sw

##取出最大d个特征值对应的特征向量，求出投影矩阵W
eigvals,eigvectors=np.linalg.eig(np.linalg.inv(sw).dot(sb))
W=np.c_[eigvectors[:,0],eigvectors[:,1]]##该死！特征向量是按列排列的！！！查了个把小时！！
X_condensed=X_train.dot(W)

##绘图同时求出各类均值及协方差
index,u,sigma=[],[],[]
xianyan=[]
plt.figure(1)
for color,i in zip('rgb',[0,1,2]):
      index=np.where(y_train==i)[0]
      plt.scatter(X_condensed[index,0],X_condensed[index,1],c=color)
      u.append(X_condensed[index,:].mean(axis=0))
      sigma.append(np.cov(X_condensed[index,:],rowvar=False))
      xianyan.append(len(index))

##预测
X_test_condensed=X_test.dot(W)
y_pred=[]
for x_test in X_test_condensed:
      temp=[]
      for i in range(3):
            temp.append(multivariate_normal.pdf(x_test,u[i],sigma[i])*xianyan[i])
      y_pred.append(temp.index(max(temp)))
print(accuracy_score(y_test,y_pred))

##sklearn的预测
lda = LinearDiscriminantAnalysis(n_components=2)
X_sklearn=lda.fit_transform(X_train,y_train)
print(lda.score(X_test,y_test))

##sklearn的图
for color,i, in zip('cmk',[0,1,2]):
      index=y_train==i
      plt.scatter(X_sklearn[index,0],X_sklearn[index,1],c=color)
plt.savefig('LDA for 3-Classification.png')

##PCA效果图
values,vectors=np.linalg.eig(X.T.dot(X))
W=vectors[:,:2]  ##投影矩阵
newX=X.dot(W)
plt.figure(2)
plt.scatter(newX[:,0],newX[:,1],s=10,c=y,cmap=plt.cm.RdYlBu)



plt.show()


















