import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,datasets

##最小二乘法解单属性输入
X,y,w0=datasets.make_regression(n_samples=100,n_features=1,n_informative=1,noise=10,
                             coef=True,random_state=19901120,bias=10) ##w0就是原始数据的系数
##X.shape=(n_samples,n_features)
x=X.reshape(y.shape)
w=np.dot(x-x.mean(),y)/(np.dot(x,x)-x.sum()**2/len(x))
b=(y-w*x).sum()/len(x)
print('原始系数=',w0,'\n模型系数=',w,'\n原始偏置=',10,'\n模型偏置=',b)
xx=np.linspace(-3,3)
yy=w*xx+b
plt.scatter(x,y)
plt.plot(xx,yy,linewidth=1,color='red',label='LSM')

lr=linear_model.LinearRegression()
lr.fit(X,y)
##[:, np.newaxis]将(n,)转化为(n,1),这个np.newaxis就等于None
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
plt.plot(line_X,line_y,color='green',label='Linear regressor')
plt.legend(loc='lower right')  ##有lebel才能生成图例

plt.show()

