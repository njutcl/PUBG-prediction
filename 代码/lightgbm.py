import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

train = pd.read_csv("train.csv")
testX = pd.read_csv("test_1.csv")
df_sample = train
df = df_sample.drop(["winPlacePerc"], axis=1)
y = df_sample['winPlacePerc']

def split_vals(a, n : int):
    return a[:n].copy(), a[n:].copy()
val_perc = 0.2
n_valid = int(val_perc * 98086)
n_trn = len(df)-n_valid
raw_train, raw_test = split_vals(df_sample, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

def lgbModel(X_train, y_train, X_valid, y_valid, testX):
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':200,
              'early_stopping_rounds':20, "num_leaves" : 30, "learning_rate" : 0.1,
              "bagging_fraction" :0.5 , "bagging_seed" : 0, "num_threads" : 8,
              "colsample_bytree" : 0.7
             }
    lgbTrain = lgb.Dataset(X_train, label=y_train)
    lgbVal = lgb.Dataset(X_valid, label=y_valid)
    model = lgb.train(params, lgbTrain, valid_sets=[lgbTrain, lgbVal],
                      early_stopping_rounds=200, verbose_eval=1000)
    yPredTest = model.predict(testX, num_iteration=model.best_iteration)
    return yPredTest, model

yPred, model = lgbModel(X_train, y_train, X_valid, y_valid, testX)

df_test = pd.read_csv('test_1.csv')
df_test['winPlacePerc'] = yPred
submission = df_test['winPlacePerc']
submission.to_csv('submission.csv', index=False)
print("submission:",submission)


#rmae评估
e1=mean_squared_error(y_true=y_valid, y_pred=yPred)
print('mean_squared_error: ', e1)

#R²评估
r2=1-(d1)/(np.std(y_valid))
print(r2_score(y_valid,yPred))
