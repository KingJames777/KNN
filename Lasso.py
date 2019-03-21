from numpy import *
from sklearn import datasets
from sklearn.linear_model import LinearRegression,Lasso

def Lasso1(X,y):
      m,n=shape(X)
      eta=1e-3
      lamda=0.5
      iterations=10000
      w=zeros(n)
      while iterations>0:
            iterations-=1
            i=random.choice(range(m),size=15,replace=False)
            gra=(X[i].T.dot(X[i]).dot(w)-X[i].T.dot(y[i]))/len(i)
            gra+=lamda*sign(w)
            w-=eta*gra
      pred=X.dot(w)
      return w,pred, corrcoef(pred,y)

def Lasso2(X,y):
      m,n=shape(X)
      error=inf
      w_best=ones(n)
      iterations=40000
      eta=1e-2
      while iterations>0:
            iterations-=1
            for i in range(n):
                  for sign in [-1,1]:
                        w_new=w_best.copy()
                        w_new[i]+=sign*eta
                        error1=sum(abs((X.dot(w_new)-y)))  ##误差函数写错，忘加绝对值！
                        if error1<error:
                              error=error1
                              w_best=w_new
      return w_best
      
if __name__=='__main__':
      X,y,=datasets.make_regression(n_samples=100,n_features=10,n_informative=4,noise=5,bias=3)
      X=(X-X.mean(axis=0))/X.std(axis=0)
      X=append(X,ones(len(X))[:,newaxis],axis=1)
      lr=Lasso(fit_intercept=False).fit(X,y)
      print(around(lr.coef_))
      print(around(Lasso1(X,y)[0]))
      print(around(Lasso2(X,y)))












































