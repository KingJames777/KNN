from sklearn.model_selection import train_test_split as tts
from Linear_Regression import R2_score
from sklearn import datasets
from numpy import *

def weight(X,x,k):
      return exp(-0.5*(sum((X-x)**2,axis=1))/k)

def LWLR(X,y,X_test,k=1):
      m,n=shape(X)
      pred=[]
      for i in range(len(X_test)):
            A=weight(X,X_test[i],k)
            temp=(X.T*A).dot(X)
            temp=linalg.inv(temp.T.dot(temp)).dot(temp.T)  ##伪逆
            w=temp.dot(X.T*A).dot(y)
            pred.append(X_test[i].dot(w))
      return pred

def kNN(X,y,X_test,k=10):
      pred=[]
      for i in range(len(X_test)):
            A=weight(X,X_test[i],1)
            res=[]
            for j in range(k):
                  index=A.argmax()
                  res.append(y[index])
                  delete(A,index)
            pred.append(sum(res)/k)
      return pred

if __name__=='__main__':
      X,y,coef=datasets.make_regression(n_samples=1000,n_features=10,n_informative=4,
                                        coef=True,noise=5,bias=3)
      coef=append(coef,3)
      X=(X-X.mean(axis=0))/X.std(axis=0)
      X=append(X,ones(len(X))[:,newaxis],axis=1)
      X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901120)

##      pred=kNN(X_train,y_train,X_test,k=20)  ##太差

      pred=LWLR(X_train,y_train,X_test)
      print('R2: ',R2_score(pred,y_test))
      print('Corrcoef: ',corrcoef(pred,y_test))






















