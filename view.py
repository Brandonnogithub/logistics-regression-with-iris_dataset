import matplotlib.pyplot as plt
import numpy as np


def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# x = np.arange(-10, 10, 0.1)
# h = Sigmoid(x)  # Sigmoid函数
# plt.plot(x, h)
# plt.axvline(0.0, color='k')
# plt.axhline(y=0.5, ls='dotted', color='k')
# plt.yticks([0.0,  0.5, 1.0])  # y axis label
# plt.title(r'Sigmoid', fontsize = 15)
# plt.text(5,0.8,r'$y = \frac{1}{1+e^{-z}}$', fontsize = 18)
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.decomposition import PCA

from plotly.offline import init_notebook_mode, iplot

iris_path = 'iris.csv'
data = pd.read_csv(iris_path)

print(data.head())

# labels = data.groupby('Species').size().index
# values = data.groupby('Species').size()
# trace = go.Pie(labels=labels, values=values)
# layout = go.Layout(width=350, height=350)
# fig = go.Figure(data=[trace], layout=layout)
# iplot(fig)

# groups = data.groupby(by = "Species")
# means, sds = groups.mean(), groups.std()
# means.plot(yerr = sds, kind = 'bar', figsize = (9, 5), table = True)
# plt.show()

# col_map = {'setosa': 'orange', 'versicolor': 'green', 'virginica': 'pink'}
# pd.tools.plotting.scatter_matrix(data.loc[:, 'Sepal.Length':'Petal.Width']
# , diagonal = 'kde', color = [col_map[lb] for lb in data['Species']], s = 75, figsize = (11, 6))
# plt.show()

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data[:, :2]             # 取前两列数据
Y = iris.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# trace = go.Scatter(x = X[:,0], y = X[:,1], mode = 'markers', 
#                     marker = dict(color = np.random.randn(150),size = 10, colorscale='Viridis',showscale=False))
# layout = go.Layout(title = '训练点', xaxis=dict(title='花萼长度 Sepal length', showgrid=False),
#                     yaxis=dict(title='花萼宽度 Sepal width',showgrid=False),
#                     width = 700, height = 380)
# fig = go.Figure(data=[trace], layout=layout)

# iplot(fig)

from sklearn.linear_model import LogisticRegression
from LogitRegression import LogitRegression_Li

# lr = LogisticRegression(C = 1e5) # C: Inverse of regularization strength
# lr = LogisticRegression(penalty='l2',solver='newton-cg',multi_class='multinomial')
# lr = LogisticRegression(penalty='l2',solver='sag',multi_class='multinomial', max_iter=100)
lr = LogitRegression_Li(n_class=3)
lr.fit(x_train,y_train)

print("Logistic Regression模型训练集的准确率：%.3f" %lr.score(x_train, y_train))
print("Logistic Regression模型测试集的准确率：%.3f" %lr.score(x_test, y_test))

from sklearn import metrics
y_hat = lr.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_hat) #错误率，也就是np.average(y_test==y_pred)
print("Logistic Regression模型正确率：%.3f" %accuracy)

target_names = ['setosa', 'versicolor', 'virginica']
print(metrics.classification_report(y_test, y_hat, target_names = target_names))


x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5 # 第0列的范围
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5 # 第1列的范围
h = .02
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h)) # 生成网格采样点

grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
grid_hat = lr.predict(grid_test)                  # 预测分类值
# grid_hat = lr.predict(np.c_[x1.ravel(), x2.ravel()])
grid_hat = grid_hat.reshape(x1.shape)             # 使之与输入的形状相同

plt.figure(1, figsize=(6, 5))
# 预测值的显示, 输出为三个颜色区块，分布表示分类的三类区域
plt.pcolormesh(x1, x2, grid_hat,cmap=plt.cm.Paired) 

# plt.scatter(X[:, 0], X[:, 1], c=Y,edgecolors='k', cmap=plt.cm.Paired)
plt.scatter(X[:50, 0], X[:50, 1], marker = '*', edgecolors='red', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], marker = '+', edgecolors='k', label='versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1], marker = 'o', edgecolors='k', label='virginica')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc = 2)

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.title("Logistic Regression", fontsize = 15)
plt.xticks(())
plt.yticks(())
plt.grid()

plt.show()