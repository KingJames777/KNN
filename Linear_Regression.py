from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression,LogisticRegression
from Lasso import Lasso1
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import datasets
from numpy import *

def linear_regression(X,y):
      w=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
      pred=X.dot(w)
      return w,pred, corrcoef(pred,y)

def ridge(X,y,lamda=1):
      w=linalg.inv(X.T.dot(X)+lamda*eye(shape(X)[1])).dot(X.T).dot(y)
      pred=X.dot(w)
      return w,pred, corrcoef(pred,y)

def R2_score(pred,y):
      u=((y - pred) ** 2).sum() 
      v=((y- y.mean(axis=0)) ** 2).sum()
      return 1-u/v

def loss_curve(X,y):
      m,n=X.shape
      w=ones(n)
      iterations=1000
      eta=0.01
      losses=[]
      while iterations>0:
            iterations-=1
            i=random.randint(m)
            w=w-eta*(w.T.dot(X[i])-y[i])*X[i]
            losses.append((X.dot(w)-y).dot(X.dot(w)-y)/m)
      x=arange(len(losses))
      plt.plot(x,losses)
      plt.show()

##多目标变量
def multi_target():
      ll=datasets.load_linnerud()
      X=ll.data
      X=append(X,ones((shape(X)[0],1)),axis=1)
      y=ll.target
      W=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
      print('预测：',X.dot(W))
      print(R2_score(X.dot(W),y))
      lr=LinearRegression().fit(X,y)
      print(lr.score(X,y))

def load_data(filename):
      n=len(open(filename).readline().split('\t'))-1  ##属性数，即列数
      X=[];y=[]
      skip=['1','2','24','25','26','27','29']
      for line in open(filename).readlines():
            lineArr=[]
            curLine=line.strip().split('\t')
            if curLine[-1] in skip:
                  continue
            for i in range(n):
                  lineArr.append(float(curLine[i]))
            X.append(lineArr)
            y.append(int(curLine[-1]))
      return array(X),array(y)

def classify(X,y):
      X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901120,stratify=y)
      lr=LogisticRegression(C=20).fit(X_train,y_train)
      print(lr.predict(X_train)[30:50],'\n',y_train[30:50])

def pca(X,y):
      pca=PCA(n_components=3)
      XT=pca.fit_transform(X)
      XT=append(XT,ones(len(X))[:,newaxis],axis=1)
##      plt.scatter(XT[:,0],XT[:,1],c=y,cmap=plt.cm.RdYlBu)
##      plt.show()
      w, pred, corrco=linear_regression(XT,y)
      print(around(pred[30:50]),'\n',y[30:50])

if __name__=='__main__':
      filename='abalone.txt'
      X,y=load_data(filename)
##      X,y,=datasets.make_regression(n_samples=100,n_features=10,random_state=1,
##                                    n_informative=6,noise=5,bias=3)
##      
##      index=[0,1,2,3,7,9]  ##系数为0的特征确实可有可无.
##      X=X[:,index]
##      pca(X,y)
      

      X=(X-X.mean(axis=0))/X.std(axis=0)
      X=append(X,ones(len(X))[:,newaxis],axis=1)
      
      coef,pred,corr=linear_regression(X,y)  ##线性预测很差
      print(around(pred[:20]),y[:20])
      
      
      
      
      












