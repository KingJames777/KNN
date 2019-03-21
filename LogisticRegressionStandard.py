from LogisticRegression import sigmoid, plotloss
import matplotlib.pyplot as plt
from numpy import *

class LogisticRegression:
      def __init__(self,max_iter=2000,tol=0.000001,C=0.2,penalty='l2'):
            self.max_iter=max_iter
            self.tol=tol
            self.C=C
            self.penalty='l2'
            self.w=array([])
            self.thre=0.5

      def loss(self,X,y,w):
            prob=sigmoid(X.dot(w))
            a=1e-12  ##防止报错
            if self.penalty=='l2':
                  return abs(-(log(prob+a).dot(y)+(1-y).dot(log(1-prob+a)))+0.5*self.C*w[:-1].dot(w[:-1]))
            else:
                  return abs(-(log(prob).dot(y)+(1-y).dot(log(1-prob))))

      def update(self,x,y,w,learning_rate):
            prob=sigmoid(x.dot(w))
            if self.penalty=='l2':
                  item1=learning_rate*(prob-y)*x
                  item2=learning_rate*self.C*w  ##这里也要学习率...
                  item2[-1]=0
                  return w-item1-item2
            else:
                  return w-learning_rate*(prob-y)*x
      
      def fit(self,X,y):
            X=c_[X,ones(X.shape[0])[:,None]]
            m,n=X.shape
            w=zeros(n)
            learning_rate=0.001
            loss_old=self.loss(X,y,w)
            count=0
            losses=[]
            losses.append(loss_old)

            nums,counts=unique(y,return_counts=True)
            self.thre=counts[1]/m
            
            while count<=self.max_iter:
                  count+=1
                  i=random.randint(m)
                  temp=self.update(X[i],y[i],w,learning_rate)
                  loss_new=self.loss(X,y,temp)
                  if abs(loss_old-loss_new)<=self.tol:
                        break
                  loss_old=loss_new
                  losses.append(loss_old)
                  w=temp
##            plotloss(losses)
            self.w=w

      def predict(self,X):
            X=c_[X,ones(X.shape[0])[:,None]]
            m,n=X.shape
            pred=[]
            for i in range(m):
                  pred.append(1) if sigmoid(X[i].dot(self.w))>=self.thre else pred.append(0)
            return pred

      ##概率图
      def prob_figure(self,X,y):
            X=c_[X,ones(X.shape[0])[:,None]]
            m=shape(X)[0]
            haxis=array(range(m))
            vaxis=sigmoid(X.dot(self.w))
            plt.scatter(haxis,vaxis,c=y,cmap=plt.cm.RdYlBu)
            plt.hlines(0.5,0,m)
            plt.show()






















