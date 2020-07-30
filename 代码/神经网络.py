import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time
import csv
from sklearn.neural_network import MLPRegressor
start = time.perf_counter()
train = pd.read_csv('train.csv')
X = train.drop(columns=['winPlacePerc'])
Y = train['winPlacePerc']
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2)
clf = MLPRegressor(hidden_layer_sizes=(300, 200, 100, 50, ), activation='relu',
                   solver='adam', alpha=0.0001, batch_size=100,  learning_rate='constant',
                   learning_rate_init=0.001, max_iter=200, shuffle=True, verbose=True,
                   early_stopping=True, validation_fraction=0.2)
clf.fit(X_train, y_train)
print("训练集准确率：",clf.score(X_train, y_train))
clf.fit(X_test, y_test)
y_Pred = clf.predict(X_test)
y_Pred[y_Pred > 1] = 1
y_Pred[y_Pred < 0] = 0

# 模型评分
y_pre = clf.predict(X_test)
c1=clf.score(X_test, y_test)
print("测试集准确率：",c1)

# mae评估
d1=mean_absolute_error(y_true=y_test, y_pred=y_Pred)
print("mae:",d1)

#rmae评估
e1=mean_squared_error(y_true=y_test, y_pred=y_Pred)
print('mean_squared_error: ', e1)

#R²评估
r2=1-(d1)/(np.std(y_test))
print("R²:",r2_score(y_test,y_Pred))

#计算时间
end=time.perf_counter()
print("final is in ",end-start)







