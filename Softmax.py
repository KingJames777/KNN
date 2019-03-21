from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
from sklearn import datasets
from numpy import *

def newy(y):
      k=len(set(y))
      m=len(y)
      new=zeros((m,k),dtype='int8')
      for i in range(k):
            temp=zeros(k,dtype='int8')
            temp[i]=1
            new[y==i]+=temp
      return new

def softmax(W,x,j):
      return exp(W[:,j].dot(x))/sum(exp(W.T.dot(x)))

def SGD(X,y):
      m,n=shape(X)
      k=shape(y)[1]
      W=zeros((n,k))
      iterations=10000
      eta=0.1
      while iterations>0:
            iterations-=1
            i=random.randint(m)
            for j in range(k):  ## 所有权向量都要更新
                  W[:,j]-=(softmax(W,X[i],j)-y[i][j])*X[i]
      return W

def predict(X,W):
      return argmax(X.dot(W),axis=1)


wine=datasets.load_wine()
X=wine.data
X=(X-X.mean(axis=0))/X.std(axis=0)  ##必须归一化，否则softmax会溢出
X=c_[X,ones(X.shape[0])[:,None]]
y=wine.target
X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,random_state=1990,stratify=y)

sm=lr(multi_class='multinomial',solver='sag')
sm.fit(X_train,y_train)
print('sklearn:',accuracy_score(y_test,sm.predict(X_test)))


y_train=newy(y_train)
W=SGD(X_train,y_train)
print(accuracy_score(predict(X_test,W),y_test))

























