# 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import linear_model # 引入线性回归方法
from sklearn.preprocessing import PolynomialFeatures
from pygam import LinearGAM
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
"""
首先，我们先来看看有监督学习中回归的例子，
我们使用sklearn内置数据集Boston房价数据集。
sklearn中所有内置数据集都封装在datasets对象内： 返回的对象有：
data:特征X的矩阵(ndarray)
target:因变量的向量(ndarray)
feature_names:特征名称(ndarray)
"""
"""
一  load_boston:
各个特征的相关解释：
CRIM：各城镇的人均犯罪率
ZN：规划地段超过25,000平方英尺的住宅用地比例
INDUS：城镇非零售商业用地比例
CHAS：是否在查尔斯河边(=1是)
NOX：一氧化氮浓度(/千万分之一)
RM：每个住宅的平均房间数
AGE：1940年以前建造的自住房屋的比例
DIS：到波士顿五个就业中心的加权距离
RAD：放射状公路的可达性指数
TAX：全部价值的房产税率(每1万美元)
PTRATIO：按城镇分配的学生与教师比例
B：1000(Bk - 0.63)^2其中Bk是每个城镇的黑人比例
LSTAT：较低地位人口
Price：房价
"""
#####1 数据的加载
boston = datasets.load_boston()
X = boston.data
Y = boston.target
features = boston.feature_names
boston_data = pd.DataFrame(X,columns=features)
boston_data["Price"] = Y
# print(boston_data.head())

######2 数据的处理
# sns.scatterplot(boston_data['NOX'],boston_data['Price'],color="r",alpha=0.6)
# plt.title("Price~NOX")
# plt.show()

#####3 拟合
lin_reg = linear_model.LinearRegression() # 创建线性回归的类
lin_reg.fit(X,Y) # 输入特征X和因变量y进行训练
print("模型系数：",lin_reg.coef_)             # 输出模型的系数
print("模型得分：",lin_reg.score(X,Y))    # 输出模型的决定系数R^2

#############4 多项回归
"""
(a) 多项式回归实例介绍：
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html?highlight=poly#sklearn.preprocessing.PolynomialFeatures
sklearn.preprocessing.PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True, order='C'):

参数：
degree：特征转换的阶数。
interaction_onlyboolean：是否只包含交互项，默认False 。
include_bias：是否包含截距项，默认True。
order：str in {‘C’, ‘F’}, default ‘C’，输出数组的顺序。

它是使用多项式的方法来进行的，如果有a，b两个特征，那么它的2次多项式为（1,a,b,a^2,ab, b^2）。
PolynomialFeatures有三个参数
degree：控制多项式的度
interaction_only： 默认为False，如果指定为True，那么就不会有特征自己和自己结合的项，上面的二次项中没有a^2和b^2。
include_bias：默认为True。如果为True的话，那么就会有上面的 1那一项
"""
X_arr = np.arange(6).reshape(3,2)
print("原始的x_arr:",X_arr)

poly = PolynomialFeatures(2)
print("2次转化x:\n",poly.fit_transform(X_arr))

poly = PolynomialFeatures(interaction_only=True)
print("2次转化x:\n",poly.fit_transform(X_arr))

"""
广义可加模型GAM实际上是线性模型推广至非线性模型的一个框架，
在这个框架中，每一个变量都用一个非线性函数来代替，但是模型本身保持整体可加性。
GAM模型不仅仅可以用在线性回归的推广，还可以将线性分类模型进行推广。具体的推广形式是：
标准的线性回归模型：
"""
gam = LinearGAM().fit(boston_data[boston.feature_names],Y)
# gam.summary()

"""
决策树回归
"""
reg_tree = DecisionTreeRegressor(criterion = "mse",min_samples_leaf = 5)
reg_tree.fit(X,Y)
print(reg_tree.score(X,Y))

"""
使用SVR(支持向量机)
kernel：核函数，{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, 默认=’rbf’。(后面会详细介绍)
degree：多项式核函数的阶数。默认 = 3。
C：正则化参数，默认=1.0。(后面会详细介绍)
epsilon：SVR模型允许的不计算误差的邻域大小。默认0.1
"""
reg_svr = make_pipeline(StandardScaler(),SVR(C=1.0,epsilon=02.))
reg_svr.fit(X,Y)
print(reg_svr.score(X,Y))

"""
偏差度量的是单个模型的学习能力，而方差度量的是同一个模型在不同数据集上的稳定性
"""

"""
特征提取的实例：向前逐步回归
案例来源：https://blog.csdn.net/weixin_44835596/article/details/89763300
根据AIC准则定义向前逐步回归进行变量筛选
"""

#定义向前逐步回归函数
def forward_select(data,target):
    pass