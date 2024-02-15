from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression

digits = load_digits()
y = digits.target == 9 # 数字9作为正类

# 创建一个9:1的不平衡数据集
x_train, x_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

# 虚拟分类器：预测多数类（非9）
dummy_majority = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
pred_most_frequent = dummy_majority.predict(x_test)
print(np.unique(pred_most_frequent)) # 预测类别：[False]
print(dummy_majority.score(x_test, y_test)) # 模型精度：0.8955555555555555

# 虚拟分类器：随机预测
dummy = DummyClassifier().fit(x_train, y_train)
pred_dummy = dummy.predict(x_test)
print(dummy.score(x_test, y_test)) # 0.8355555555555556

# 真实分类器：随机预测
logreg = LogisticRegression(C=0.1).fit(x_train, y_train)
pred_logreg = logreg.predict(x_test)
print(logreg.score(x_test, y_test)) # 0.9844444444444445
