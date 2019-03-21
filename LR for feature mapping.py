from LogisticRegression import sigmoid,SGD,predict
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import *

def show_fig(X,y):
      for i,mark in zip(range(2),['x','+']):
            plt.scatter(X[y==i][:,0],X[y==i][:,1],marker=mark)
      plt.show()

def basic_func(X):  ##映射至高维
      res=[]
      m,n=shape(X)
      degree=6
      for i in range(m):
            out=[]
            for j in range(1,degree+2):
                  for k in range(j):
                        out.append(X[i][0]**k*(X[i][1]**(j-k-1)))
            res.append(out)
      return array(res)

if __name__=='__main__':
      X=[];y=[]
      filename=r'ex2data2.txt'
      file=open(filename)
      for line in file.readlines():
            if line=='\n':  ##再闹看看？
                  continue
            a=line.strip().split(',')
            X.append([float(a[0]),float(a[1])])
            y.append(int(a[2]))
            
      X=array(X)
      y=array(y)
      X=basic_func(X)
      X=c_[X,ones(X.shape[0])[:,None]]
      X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901120,stratify=y)
      w=SGD(X_train,y_train)
      print(accuracy_score(y_test,predict(X_test,w)))
      







