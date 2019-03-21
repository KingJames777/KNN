from numpy import *
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

def original(X,y):
      m,n=shape(X)
      w=ones(n)
      iterations=10000
      eta=1
      while iterations>0:
            iterations-=1
            i=random.randint(m)
            if X[i].dot(w)*y[i]<=0:
                  w=w+eta*y[i]*X[i]
      return w

def dual(X,y):
      m,n=shape(X)
      alpha=zeros(m)
      Gram=X.dot(X.T)
      iterations=10000
      eta=1
      while iterations>0:
            iterations-=1
            i=random.randint(m)
            if y[i]*((alpha*y).dot(Gram[i]))<=0:
                  alpha[i]+=eta
      print(alpha)
      return X.T.dot(alpha*y)
                  

def predict(X,w):
      return sign(X.dot(w))

clf = Perceptron(max_iter=100,eta0=0.2)

bc=datasets.load_breast_cancer()
X=bc.data
X=append(X,ones((shape(X)[0],1),dtype='int8'),axis=1)
y=bc.target
y[y==0]=-1
X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19541020,stratify=y)
w=original(X_train,y_train)
print('Original for bc:',accuracy_score(y_test,predict(X_test,w)))
w=dual(X_train,y_train)
print('Dual for bc:',accuracy_score(y_test,predict(X_test,w)))
clf.fit(X_train,y_train) 
print('sklearn for bc:',clf.score(X_test,y_test)) 

iris=datasets.load_iris()
X=iris.data
y=iris.target
index=y==2
index=logical_not(index)
X=X[index]
X=append(X,ones((shape(X)[0],1),dtype='int8'),axis=1)
y=y[index]
y[y==0]=-1
X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=19901020,stratify=y)

w=original(X_train,y_train)
print('Original for iris:',accuracy_score(y_test,predict(X_test,w)))
w=dual(X_train,y_train)
print('Dual for iris:',accuracy_score(y_test,predict(X_test,w)))
clf.fit(X_train,y_train) 
print('sklearn for bc:',clf.score(X_test,y_test)) 






