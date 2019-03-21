import numpy as np
from sklearn.datasets import make_classification,load_wine
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import random

n_classes=3
lw=load_wine()
X=lw.data
y = lw.target
##X,y=make_classification(n_samples=500,n_informative=8,n_classes=n_classes,
##                        n_clusters_per_class=1)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,
                                                  random_state=19541020,stratify=y)
lda=LinearDiscriminantAnalysis().fit(X_train,y_train)
print('未采用OVO时的准确率： ',lda.score(X_test,y_test))

X_ovo,y_ovo=[],[]
for i in range(n_classes-1):
      for j in range(i+1,n_classes):
            index=np.logical_or(y_train==i,y_train==j)
            X_ovo.append(X_train[index])
            y_ovo.append(y_train[index])

n_classifiers=int(n_classes*(n_classes-1)/2)
predictions=np.empty([n_classifiers,X_test.shape[0]])
for i in range(n_classifiers):
      lda=LinearDiscriminantAnalysis().fit(X_ovo[i],y_ovo[i])
      predictions[i]=lda.predict(X_test)
predictions=predictions.astype(np.int16)

real_prediction=[]
for i in range(X_test.shape[0]):
      real_prediction.append(np.argmax(np.bincount(predictions[:,i])))
print('采用OVO的准确率： ',accuracy_score(real_prediction,y_test))

y_ovm=[]
for i in range(n_classes):
      y_temp=y_train
      index=np.where(y_train==i)
      y_temp=np.zeros(len(y_train))
      y_temp[index]=1
      y_ovm.append(y_temp)

n_classifiers=n_classes
predictions=np.empty([n_classifiers,X_test.shape[0]])
for i in range(n_classifiers):
      lda=LinearDiscriminantAnalysis().fit(X_train,y_ovm[i])
      predictions[i]=lda.predict(X_test)

real_prediction=[]
for i in range(X_test.shape[0]):
      ssum=sum(predictions[:,i])
      if ssum==0:
            real_prediction.append(np.random.randint(n_classifiers))
      elif ssum==1:
            real_prediction.append(np.argmax(predictions[:,i]))
      else:
            index=np.where(predictions[:,i]==1)
            real_prediction.append(random.choice(index[0]))

print('采用OVR的准确率： ',accuracy_score(real_prediction,y_test))  ##奇差！！





















