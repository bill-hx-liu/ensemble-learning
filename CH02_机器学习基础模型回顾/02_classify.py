import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

"""
各个特征的相关解释：

sepal length (cm)：花萼长度(厘米)
sepal width (cm)：花萼宽度(厘米)
petal length (cm)：花瓣长度(厘米)
petal width (cm)：花瓣宽度(厘米)
"""

iris = datasets.load_iris()
X = iris.data
Y = iris.target
features = iris.feature_names
iris_data = pd.DataFrame(X,columns=features)
iris_data["target"] = Y
print(iris_data.head())

# 可视化特征
marker = ['s','x','o']
for index,c in enumerate(np.unique(Y)):
    plt.scatter(x=iris_data.loc[Y==c,"sepal length (cm)"],y=iris_data.loc[Y==c,"sepal width (cm)"],alpha=0.8,label=c,marker=marker[c])
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()
plt.show()
