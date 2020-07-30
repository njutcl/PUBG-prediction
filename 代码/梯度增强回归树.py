from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#自定义函数，分割训练集和验证集
train = pd.read_csv("D:/something/homework/san2/train.csv")
df_sample = train
df = df_sample.drop(["winPlacePerc"], axis=1)
y = df_sample['winPlacePerc']

def split_vals(a, n : int):
    return a[:n].copy(), a[n:].copy()
val_perc = 0.2
n_valid = int(val_perc * 98086)
n_trn = len(df)-n_valid

raw_train, raw_valid = split_vals(df_sample, n_trn)
X_train, X_test = split_vals(df, n_trn)
y_train, y_test = split_vals(y, n_trn)

#构建模型
GBR = GradientBoostingRegressor(loss='ls',learning_rate=0.1,
                                n_estimators=100,max_depth=3)
GBR.fit(X_train,y_train)

#模型可视化
y_pred_train = GBR.predict(X_train)
y_pred_test = GBR.predict(X_test)

y_pred_train[y_pred_train>1] = 1
y_pred_train[y_pred_train<0] = 0

f, ax = plt.subplots(figsize=(10,10))
plt.scatter(y_train, y_pred_train)
plt.xlabel("y")
plt.ylabel("y_pred_train")
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()

#预测准确率
GBR_SCORE=GBR.score(X_train,y_train)
print('GBR_SCORE: ', GBR_SCORE)

# mae评估
y_pre = GBR.predict(X_test)
d1=mean_absolute_error(y_true=y_test, y_pred=y_pre)
print('mean_absolute_error: ', d1)

#rmae评估
e1=mean_squared_error(y_true=y_test, y_pred=y_pre)
print('mean_squared_error: ', e1)

#R²评估
r2=1-(d1)/(np.std(y_test))
print("R²：",r2_score(y_test,y_pre))
