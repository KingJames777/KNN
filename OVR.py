from sklearn.model_selection import train_test_split as tts
from LogisticRegressionStandard import LogisticRegression
##from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
from os import listdir
from numpy import *

##最终成绩92%，不如调用sklearn的96%，问题还是出在LR本身，OVR并没问题.

def load_data():
      X=[]; y=[]
      folder=['trainingDigits','testDigits']
      for name in folder:
            filelist=listdir(name)
            for filename in filelist:
                  X.append(img2vector(name+'\%s'%filename))
                  y.append(int(filename[0]))
      return X,y

def img2vector(filename):
      res=[]
      file=open(filename)
      for i in range(32):
            line=file.readline()
            for j in range(32):
                  res.append(int(line[j]))
      return res

##训练k个分类器
def OVR(X,X_test,y,y1,k):
      Cs=[]
      for i in range(k):
            y_train=y.copy()  ##  y不动！
            y_test=y1.copy()
            
            index=y_train==i
            y_train[index]=1  ##比如5号分类器对于数字5才会输出1，否则输出0
            y_train[logical_not(index)]=0
            
##            index=y_test==i
##            y_test[index]=1  ##比如5号分类器对于数字5才会输出1，否则输出0
##            y_test[logical_not(index)]=0
##            
            lr=LogisticRegression(max_iter=2000,tol=0.000001,C=0.2)
            lr.fit(X,y_train)
##            pred=lr.predict(X_test)
##            print(accuracy_score(y_test,pred))
            Cs.append(lr)
      return Cs

def select_best(Classifiers,X,index):
      prob=-inf;  maxn=inf
      for n in index:  ##  w.x最大者对应的概率最大
            p=Classifiers[n].w.dot(X)
            if p>prob:
                  maxn=n
                  prob=p
      return maxn

if __name__=='__main__':
      X,y=load_data()
      X=array(X);   y=array(y);  k=10
      acc,Classifiers=[],[]
      for i in range(10):
            pred,final_pred=[],[]
            X_train, X_test, y_train, y_test=tts(X,y,test_size=0.2,
                                                 random_state=random.randint(1,100),stratify=y)
            
            Classifiers=OVR(X_train,X_test,y_train,y_test,k)
            for j in range(k):
                  pred.append(Classifiers[j].predict(X_test))   ##这里就不对，问题在于OVR函数！
            pred=array(pred)
            
            for j in range(len(X_test)):  ##逐列查看
                  temp,count=unique(pred[:,j],return_counts=True)
                  temp_X=append(X_test[j],1)
                  if 1 not in temp:  ##没一个分类器预测正确
                        final_pred.append(select_best(Classifiers,temp_X,range(k)))
                  elif count[1]>1:  ##不止一个分类器预测为1
                        index=where(pred[:,j]==1)  ##预测为1的分类器索引
                        final_pred.append(select_best(Classifiers,temp_X,index[0]))
                  else:  ##唯一一个输出1
                        final_pred.append(argmax(pred[:,j]))
            print(y_test[:20],final_pred[:20])
            acc.append(accuracy_score(y_test,final_pred))
      print(acc,'\n',mean(acc))














