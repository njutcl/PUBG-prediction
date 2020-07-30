import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
train = pd.read_csv('train_V2.csv')
# coding: utf-8

#查空值
def LOST(train):
    print(train[train['Id'].isnull()])
    print(train[train['groupId'].isnull()])
    print(train[train['matchId'].isnull()])
    print(train[train['assists'].isnull()])
    print(train[train['boosts'].isnull()])
    print(train[train['damageDealt'].isnull()])
    print(train[train['DBNOs'].isnull()])
    print(train[train['headshotKills'].isnull()])
    print(train[train['heals'].isnull()])
    print(train[train['killPlace'].isnull()])
    print(train[train['killPoints'].isnull()])
    print(train[train['kills'].isnull()])
    print(train[train['killStreaks'].isnull()])
    print(train[train['longestKill'].isnull()])
    print(train[train['matchDuration'].isnull()])
    print(train[train['maxPlace'].isnull()])
    print(train[train['numGroups'].isnull()])
    print(train[train['rankPoints'].isnull()])
    print(train[train['revives'].isnull()])
    print(train[train['rideDistance'].isnull()])
    print(train[train['roadKills'].isnull()])
    print(train[train['swimDistance'].isnull()])
    print(train[train['teamKills'].isnull()])
    print(train[train['vehicleDestroys'].isnull()])
    print(train[train['walkDistance'].isnull()])
    print(train[train['weaponsAcquired'].isnull()])
    print(train[train['winPoints'].isnull()])
    print(train[train['winPlacePerc'].isnull()])

#输入标准化
def standard(train):
    count = train.groupby('matchId')['matchId'].transform('count')
    train['numjoined'] = count
    train.drop(train[train['numjoined'] <= 80].index, inplace=True)
    train['kills'] = train['kills'] * ((100 - train['numjoined']) / 100 + 1)
    train['damageDealt'] = train['damageDealt'] * ((100 - train['numjoined']) / 100 + 1)
    train['maxPlace'] = train['maxPlace'] * ((100 - train['numjoined']) / 100 + 1)
    train['matchDuration'] = train['matchDuration'] * ((100 - train['numjoined']) / 100 + 1)
    print(train)
    return(train)

#绘制条形图，可以根据特征改变
def pictures(columns):
    plt.figure(figsize=(10, 4))
    sns.countplot(data=train, x=columns).set_title('columns')
    plt.show()

#去掉异常值
def abnormal(train):
    train.drop(train[train['kills'] > 30].index, inplace=True)
    train['headshot_rate'] = train['headshotKills'] / train['kills']
    train['headshot_rate'] = train['headshot_rate'].fillna(0)
    train.drop(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].index, inplace=True)
    train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
    train['distance']=train['rideDistance']+train['swimDistance']+train['walkDistance']
    train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)
    train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
    train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
    train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
    train.drop(train[train['heals'] >= 40].index, inplace=True)
    train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['distance'] == 0))
    train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
    print(train)
    return (train)

#相关性计算函数
def corr(vector_A, vector_B):
    if vector_A.shape[0] != vector_B.shape[0]:
        raise Exception('The Vector must be the same size')
    vector_A_mean, vector_B_mean = np.mean(vector_A), np.mean(vector_B)
    vector_A_diff, vector_B_diff = vector_A - vector_A_mean, vector_B - vector_B_mean
    molecule = np.sum(vector_A_diff * vector_B_diff)
    denominator = np.sqrt(np.sum(vector_A_diff ** 2) * np.sum(vector_B_diff ** 2))
    return molecule / denominator

#one hot编码
def category(train):
    train['matchType'].unique()
    train = pd.get_dummies(train, columns=['matchType'])
    print(train)
    return (train)

#相关性选择
def feature_selection(train):
    print("assists:", corr(train['assists'], train['winPlacePerc']))
    print("boosts:", corr(train['boosts'], train['winPlacePerc']))
    print("damageDealt:", corr(train['damageDealt'], train['winPlacePerc']))
    print("DBNOs:", corr(train['DBNOs'], train['winPlacePerc']))
    print("headshotKills:", corr(train['headshotKills'], train['winPlacePerc']))
    print("heals:", corr(train['heals'], train['winPlacePerc']))
    print("killPlace:", corr(train['killPlace'], train['winPlacePerc']))
    print("killPoints:", corr(train['killPoints'], train['winPlacePerc']))
    print("kills:", corr(train['kills'], train['winPlacePerc']))
    print("killStreaks:", corr(train['killStreaks'], train['winPlacePerc']))
    print("longestKill:", corr(train['longestKill'], train['winPlacePerc']))
    print("matchDuration:", corr(train['matchDuration'], train['winPlacePerc']))
    #print("matchType:", corr(train['matchType'], train['winPlacePerc']))
    print("maxPlace:", corr(train['maxPlace'], train['winPlacePerc']))
    print("numGroups:", corr(train['numGroups'], train['winPlacePerc']))
    print("rankPoints:", corr(train['rankPoints'], train['winPlacePerc']))
    print("revives:", corr(train['revives'], train['winPlacePerc']))
    print("rideDistance:", corr(train['rideDistance'], train['winPlacePerc']))
    print("roadKills:", corr(train['roadKills'], train['winPlacePerc']))
    print("swimDistance:", corr(train['swimDistance'], train['winPlacePerc']))
    print("teamKills:", corr(train['teamKills'], train['winPlacePerc']))
    print("vehicleDestroys:", corr(train['vehicleDestroys'], train['winPlacePerc']))
    print("walkDistance:", corr(train['walkDistance'], train['winPlacePerc']))
    print("weaponsAcquired:", corr(train['weaponsAcquired'], train['winPlacePerc']))
    print("numjoined:", corr(train['numjoined'], train['winPlacePerc']))
    print("distance:", corr(train['distance'], train['winPlacePerc']))
    print("winPoints:", corr(train['winPoints'], train['winPlacePerc']))
    print("squad-fpp:", corr(train['matchType_squad-fpp'], train['winPlacePerc']))
    print("duo:", corr(train['matchType_duo'], train['winPlacePerc']))
    print("squad:", corr(train['matchType_squad'], train['winPlacePerc']))
    print("duo-fpp:", corr(train['matchType_duo-fpp'], train['winPlacePerc']))
    print("solo:", corr(train['matchType_solo'], train['winPlacePerc']))
    #print("normal-squad-fpp:", corr(train['matchType_normal-squad-fpp'], train['winPlacePerc']))
    print("crashfpp:", corr(train['matchType_crashfpp'], train['winPlacePerc']))
    #print("normal-solo-fpp:", corr(train['matchType_normal-solo-fpp'], train['winPlacePerc']))
    #print("normal-duo-fpp:", corr(train['matchType_normal-duo-fpp'], train['winPlacePerc']))
    #print("normal-solo:", corr(train['matchType_normal-solo'], train['winPlacePerc']))
    print("solo-fpp:", corr(train['matchType_solo-fpp'], train['winPlacePerc']))

#主函数
def main(train):
    LOST(train)
    train=standard(train)
    train=abnormal(train)
    train=category(train)
    feature_selection(train)
    #train=delete(train)
    return(train)

train=main(train)
#删除不相关数据项
train=train.drop(columns=['Id','groupId','matchId','killPoints','matchDuration','maxPlace','numGroups','rankPoints',
                        'roadKills','teamKills','vehicleDestroys','numjoined','winPoints','matchType_squad-fpp',
                        'matchType_duo','matchType_duo-fpp','matchType_solo','matchType_crashfpp','matchType_solo-fpp',
                        'killsWithoutMoving','matchType_crashfpp','matchType_duo','matchType_duo-fpp','matchType_solo',
                        'matchType_solo-fpp','matchType_squad','matchType_squad-fpp','headshot_rate'])
#保存文件
outputpath='train.csv'
train.to_csv(outputpath,sep=',',index=False,header=True)