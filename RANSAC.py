from numpy import *
from sklearn import linear_model, datasets

def LSM(X,y):
      return linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def RANSAC(X,y):
      m=len(y)
      n_consensus=int(m*0.3)  ##有这么多假定的inlier才算合格
      n_start=shape(X)[1]+1  ##初始值不能太大，以免一开始就把outlier抓进来！
      iterations=2000 ##迭代轮次
      threhold=100
      best_w=LSM(X,y)
      best_consensus_set=[]
      best_error=inf
      while iterations>0:
            maybe_inliers = list(random.choice(range(m),size=n_start,replace=False))
            w=LSM(X[maybe_inliers],y[maybe_inliers])
            maybe_outliers=set(range(m))-set(maybe_inliers)
            for i in maybe_outliers:
                  if abs(X[i].dot(w)-y[i])<=threhold:
                        maybe_inliers.append(i)
            n_maybe_inliers=len(maybe_inliers)
            if n_maybe_inliers>=n_consensus:  ##达标
                  w=LSM(X[maybe_inliers],y[maybe_inliers])
                  this_error=sum(abs(X[maybe_inliers].dot(w)-y[maybe_inliers]))/n_maybe_inliers
                  if this_error<best_error:
                        best_error=this_error
                        best_w=w
                        best_consensus_set=maybe_inliers
            iterations-=1
      return best_w,best_consensus_set

n_samples = 1000
n_outliers = 100
n_features=10
n_inform=4
X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=n_features,
                                      n_informative=n_inform, noise=10,coef=True, random_state=0,bias=5)
random.seed(0)
X[:n_outliers] = 3 + 0.5 * random.normal(size=(n_outliers, n_features)) 
y[:n_outliers] = -3 + 10 * random.normal(size=n_outliers)
X=append(X,ones(n_samples)[:,newaxis],axis=1)

w,inliers=RANSAC(X,y)
print('实验结果:',around(w),'\n',len(inliers))
print('真实结果:',around(coef))

ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
print('别人结果:',around(LSM(X[inlier_mask],y[inlier_mask])))



