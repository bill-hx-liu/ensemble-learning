"""
投票法的局限性在于，它对所有模型的处理是一样的，
这意味着所有模型对预测的贡献是一样的。如果一些模型在某些情况下很好，而在其他情况下很差，
这是使用投票法时需要考虑到的一个问题。

投票法的案例分析(基于sklearn，介绍pipe管道的使用以及voting的使用)
  Sklearn中提供了 VotingRegressor 与 VotingClassifier 两个投票方法。   
这两种模型的操作方式相同，并采用相同的参数。使用模型需要提供一个模型列表，
列表中每个模型采用Tuple的结构表示，第一个元素代表名称，第二个元素代表模型，
需要保证每个模型必须拥有唯一的名称。
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

"""
有时某些模型需要一些预处理操作，我们可以为他们定义Pipeline完成模型预处理工作
模型还提供了voting参数让我们选择软投票或者硬投票：
"""
models = [('lr',LogisticRegression()),('svm',make_pipeline(StandardScaler,SVC()))]
ensemble = VotingClassifier(estimators=models,voting='soft')
"""
bagging的思路
与投票法不同的是，Bagging不仅仅集成模型最后的预测结果，同时采用一定策略来影响基模型训练，
保证基模型可以服从一定的假设。在上一章中我们提到，希望各个模型之间具有较大的差异性，
而在实际操作中的模型却往往是同质的，因此一个简单的思路是通过不同的采样增加模型的差异性。

bagging的原理分析
Bagging的核心在于自助采样(bootstrap)这一概念，即有放回的从数据集中进行采样，也就是说，
同样的一个样本可能被多次进行采样。一个自助采样的小例子是我们希望估计全国所有人口年龄的平均值，
那么我们可以在全国所有人口中随机抽取不同的集合（这些集合可能存在交集），计算每个集合的平均值，然后将所有平均值的均值作为估计值。
首先我们随机取出一个样本放入采样集合中，再把这个样本放回初始数据集，重复K次采样，最终我们可以获得一个大小为K的样本集合。
同样的方法， 我们可以采样出T个含K个样本的采样集合，然后基于每个采样集合训练出一个基学习器，再将这些基学习器进行结合，这就是Bagging的基本流程。
对回归问题的预测是通过预测取平均值来进行的。对于分类问题的预测是通过对预测取多数票预测来进行的。
Bagging方法之所以有效，是因为每个模型都是在略微不同的训练数据集上拟合完成的，这又使得每个基模型之间存在略微的差异，
使每个基模型拥有略微不同的训练能力。Bagging同样是一种降低方差的技术，因此它在不剪枝决策树、神经网络等易受样本扰动的学习器上效果更加明显。
在实际的使用中，加入列采样的Bagging技术对高维小样本往往有神奇的效果。

Sklearn为我们提供了 BaggingRegressor 与 BaggingClassifier 两种Bagging方法的API，
我们在这里通过一个完整的例子演示Bagging在分类问题上的具体应用。这里两种方法的默认基模型是树模型
"""


"""
下面我们使用sklearn来实现基于决策树方法的bagging策略
"""

"""
我们将使用重复的分层k-fold交叉验证来评估该模型，一共重复3次，每次有10个fold。
我们将评估该模型在所有重复交叉验证中性能的平均值和标准差
"""
# evaluate bagging algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

# define dataset
#1 创建一个含有1000个样本20维特征的随机分类数据集：
X,y = make_classification(n_samples=1000,n_features=20,n_informative=15,
                          n_redundant=5,random_state=5)
# summarize the dataset
print(X.shape,y.shape)

# define the model
model = BaggingClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=3)
n_scores = cross_val_score(model,X,y,scoring='accuracy',cv=cv,n_jobs=-1,error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

