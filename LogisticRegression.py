from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy import *

def sigmoid(x):
      return 1/(1+exp(-x))

def loss(X,y,w,lamda):
      prob=sigmoid(X.dot(w))
      a=1e-12  ##防止报错
      return abs(-(log(prob+a).dot(y)+(1-y).dot(log(1-prob+a)))+0.5*lamda*w[:-1].dot(w[:-1]))

def plotloss(losses):
      x=arange(len(losses))
      plt.plot(x,losses)
      plt.show()

def SGD(X,y):
      m,n=X.shape
      w=zeros(n)
      iterations=500
      count=iterations
      eta=0.1
      lamda=0.1
      losses=[]
      losses.append(loss(X,y,w,lamda))
      while count>0:
            count-=1
            i=random.randint(m)
            item1=eta*(sigmoid(w.dot(X[i]))-y[i])*X[i]
            item2=eta*lamda*w
            item2[-1]=0
            w=w-item1-item2
            losses.append(loss(X,y,w,lamda))
      plotloss(losses)
      return w

def GD(X,y):
      m,n=X.shape
      w=ones(n)
      iterations=100
      eta=1
      lamda=0.001
      losses=[]
      losses.append(loss(X,y,w))
      while iterations>0:
            iterations-=1
            w=w-eta*X.T.dot(sigmoid(X.dot(w))-y)-lamda*w
            losses.append(loss(X,y,w,lamda))
      plotloss(losses)
      return w

def Newton(X,y):
      m,n=X.shape
      w=random.randn(n)
      iterations=50
      lamda=0.0001
      losses=[]
      losses.append(loss(X,y,w))
      while iterations>0:
            try:
                  iterations-=1
                  temp=sigmoid(X.dot(w))
                  R=diag(temp*(1-temp)+0.0001)  ##死活遇到奇异矩阵...
                  H=X.T.dot(R).dot(X)
                  w=w-linalg.inv(H).dot(X.T).dot(temp-y)
                  losses.append(loss(X,y,w,lamda))
            except:
                  break
      plotloss(losses)
      return w

def predict(X,w):
      m,n=X.shape
      pred=[]
      for i in range(m):
            pred.append(1) if sigmoid(X[i].dot(w))>=1/2 else pred.append(0)
      return pred

if __name__=='__main__':
      bc=datasets.load_breast_cancer()
      X=bc.data
      y=bc.target
      X=(X-X.mean(axis=0))/X.std(axis=0)  ##如若不归一化，收敛速度大减，此外容易溢出。
      X=c_[X,ones(X.shape[0])[:,None]]
      X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901120,stratify=y)

      w=SGD(X_train,y_train)
      print(accuracy_score(y_test,predict(X_test,w)))
      ##w=GD(X_train,y_train)
      ##print(accuracy_score(y_test,predict(X_test,w)))
      ##w=Newton(X_train,y_train)
      ##print(accuracy_score(y_test,predict(X_test,w)))
      print(w[w>1])  ##系数确实很大,加了正则项以后小得多，可正确率却下降了？

      lrr=lr()
      lrr.fit(X_train,y_train)
      print(accuracy_score(lrr.predict(X_test),y_test))

















