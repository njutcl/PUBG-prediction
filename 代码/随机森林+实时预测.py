import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from chicken_dinner.pubgapi import PUBG
import datetime

# 阶段1
train = pd.read_csv("D:/LTC/大三下/机器学习/数据集+代码/train.csv")
df_sample = train
df_sample.shape
df = df_sample.drop(["winPlacePerc"], axis=1)
col=df.columns
y = df_sample['winPlacePerc']


#自定义函数，分割训练集和验证集
def split_vals(a, n : int):
    return a[:n].copy(), a[n:].copy()
val_perc = 0.2
n_valid = int(val_perc * 98086)
n_trn = len(df)-n_valid

raw_train, raw_valid = split_vals(df_sample, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

#检查数据集维度
print('Sample train shape: ', X_train.shape,
      '\nSample target shape: ', y_train.shape,
      '\nSample validation shape: ', X_valid.shape)

#模型运算
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

m1.fit(X_train, y_train) 
y_pre = m1.predict(X_valid)

#模型准确率
a1=m1.score(X_valid, y_valid)

#mae评估
b1=mean_absolute_error(y_true=y_valid, y_pred=y_pre)
print('m1.score: ', a1,
      '\nmean_absolute_error: ', b1)

#rmae评估
e1=mean_squared_error(y_true=y_valid, y_pred=y_pre)
print('mean_squared_error: ', e1)

#R²评估
r2=1-(b1)/(np.std(y_valid))
print('R²评估: ', r2_score(y_valid,y_pre))

#特征在当前模型的重要性
feature_importances=m1.feature_importances_
imp_df = pd.DataFrame({"cols":df.columns, "imp":m1.feature_importances_})
print(imp_df)

# 保留比较重要的特征
to_keep = imp_df[imp_df.imp>0.005].cols
print('Significant features: ', len(to_keep))

# 由这些比较重要的特征值，生成新的数据集，重新制定训练集和测试集
df_keep = df[to_keep]
X_train, X_valid = split_vals(df_keep, n_trn)

# 模型训练
m2 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',n_jobs=-1)
m2.fit(X_train, y_train)

# 决定系数
y_pre = m2.predict(X_valid)
c1=m2.score(X_valid, y_valid)
# mae评估
d1=mean_absolute_error(y_true=y_valid, y_pred=y_pre)
print('m1.score: ', c1,
      '\nmean_absolute_error: ', d1)

#rmae评估
e1=mean_squared_error(y_true=y_valid, y_pred=y_pre)
print('mean_squared_error: ', e1)

#R²评估
r2=1-(d1)/(np.std(y_valid))
print('R²评估: ', r2_score(y_valid,y_pre))

 #阶段二
 #获取数据
api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiI1ODYxYWU3MC04MDk5LTAxMzgtNGEzYS0wMDkzNGM5NGQzOGEiLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNTkwMzk5NDc5LCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6InNjdWx0Yy1nbWFpbC1jIn0.wyTw_Zm0HXcE2Vs6k1udjWO-dMerDdQQ6eyVxpx7Wkk"
pubg = PUBG(api_key, "pc-na")
# shroud = pubg.players_from_names("shroud")[0]
match_id = 'eac52389-67c2-418b-9d10-fa8c8a273dac'
match = pubg.match(match_id)
telemetry = match.get_telemetry()
print(telemetry)
numPlayers=telemetry.num_players()
print(numPlayers)

team="02824761-d815-40cf-9b5a-4465d1f7eb5e"
print(team)


# print(match.rosters[0].participants[0].player_id)
#日期格式变为秒数方便计算    
def timeFormat(utc):
    UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
    utcTime = datetime.datetime.strptime(utc, UTC_FORMAT)
    localtime = utcTime + datetime.timedelta(hours=8)
    return (localtime-datetime.datetime(1970,1,1)).total_seconds()
    
#按照与阶段一数据项的对应关系从日志数据中获取不同类型的事件
# player=team.player_ids[0]


players=['account.8319864b28e743f0b3bdff3835c95e1f','account.3bfc4e6ccba9431ebbc23b6917f8df64','account.872c676a1f1c4218aa903daf9b92cc79','account.3fe9f5f3469640b2b6bc842cfd22d49d']
number=0
player="account.3fe9f5f3469640b2b6bc842cfd22d49d"

# print(player)
# player=match.rosters[0].participants[0].name
# print(player1)
# print(telemetry.event_types())
heal=telemetry.filter_by("log_heal")
kill=telemetry.filter_by('log_player_kill')
revive=telemetry.filter_by("log_player_revive") 
damage=telemetry.filter_by("log_player_take_damage") 
DBNO=telemetry.filter_by("log_player_make_groggy")
WeaponsAcquired1=telemetry.filter_by("log_item_pickup_from_loot_box")
WeaponsAcquired2=telemetry.filter_by("log_item_pickup")
WeaponsAcquired3=telemetry.filter_by("log_item_pickup_from_carepackage")
boost=telemetry.filter_by("log_item_use")
# swimDistance=match.rosters[0].participants[0].stats['swim_distance']
swimDistance=0
# walkDistance=match.rosters[0].participants[0].stats['walk_distance']
walkDistance=  2985.5837
# rideDistance=match.rosters[0].participants[0].stats['ride_distance']
rideDistance=0
#直接用最后的结果  因为这个计算比较麻烦
killPlace=1
# 20
# killPlace=match.rosters[0].participants[0].stats['kill_place']


        #   "timeSurvived": 1555.525,
        #   "walkDistance":3203.6497 ,

# rideStart=telemetry.filter_by("log_player_vehicle_ride")
# rideLeave=telemetry.filter_by("log_player_vehicle_leave")
# swimStart=telemetry.filter_by("log_player_swim_start")
# swimEnd=telemetry.filter_by("log_player_swim_end")


# 目标玩家各类日志数据提取，只保留时间和需要的统计字段，一条事件一个tuple，存储在list内
#增益 boost
def boostInTeam(o): 
    if o["character"]['account_id'] == player and o['item']['sub_category']=='Boost':
        i=o["character"]['account_id']
        return  i


loghboost=list(filter(boostInTeam,boost))

boost_events=[]
boostCount=0
for i in loghboost:
    time=timeFormat(i.timestamp)
    boostCount+=1
    boost_events.append((time,boostCount))
    
print("boost",boost_events)


#武器获取
def weaponInTeam(o): 
    if o["character"]['account_id'] == player and o['item']['category']=='Weapon':
        i=o["character"]['account_id']
        return  i


loghweapon=list(filter(weaponInTeam,WeaponsAcquired1+WeaponsAcquired2+WeaponsAcquired3))

weapon_events=[]
weaponCount=0
for i in loghweapon:
    time=timeFormat(i.timestamp)
    weaponCount+=1
    weapon_events.append((time,weaponCount))
    
print("weapon",weapon_events)


#治疗
def healInTeam(o):
    i=o['character']['account_id']
    if i == player:
        return  i


logheal=list(filter(healInTeam,heal))

heal_events=[]
healCount=0
for i in logheal:
    time=timeFormat(i.timestamp)
    healCount+=1
    
    heal_events.append((time,healCount))
    
print("heal",heal_events)

#杀敌 助攻 爆头 最远距离 连杀（一分钟之内）   
def killInTeam(o):
    if(o['killer']): #自己死的不算
        i=o['killer']['account_id']
        if i in players:  #全队的
            return  i

logkill=list(filter(killInTeam,kill))

kill_events=[]
assist_events=[]
assist=0
killCount=0
headshotKills=0
LongestKill=0
killStreaks=0
killStreaks_tmp=0
victims=[]
for i in logkill:
    if(i['killer']['account_id']==player):
        victims.append(i['victim']['name'])
        killCount+=((100 - numPlayers) / 100 + 1)
        time=timeFormat(i.timestamp)
        lastkill=time
        if(i['damageReason']=='HeadShot'):
            headshotKills+=1
        if(i['distance']>LongestKill):
            LongestKill=i['distance']
           
        if((time-lastkill)<60):
            killStreaks_tmp+=1
            killStreaks=killStreaks_tmp if killStreaks_tmp>killStreaks else killStreaks
        else:
            killStreaks_tmp=1
        
        kill_events.append((time,killCount,headshotKills,LongestKill,killStreaks))
    if(i['assistant']['account_id']== player):
        time=timeFormat(i.timestamp)
        assist+=1
        assist_events.append((time,assist))
  
print(victims)
print("kill",kill_events)
print("assist",assist_events)
#救人
revive_events=[]
def reviveInTeam(o):
        i=o['victim']['account_id']
        if i == player:
            return  i

logrevive=list(filter(reviveInTeam,revive))

revives=0    
for i in logrevive:
    revives+=1
    time=timeFormat(i.timestamp)
    revive_events.append((time,revives))
print("revive",revive_events)

#承受伤害

Damage_events=[]
def takeDamageInTeam(o):
        i=o['victim']['account_id']
        if i == player:
            return  i

logtakedamage=list(filter(takeDamageInTeam,damage))



DamageDealt=0
for i in logtakedamage:
    time=timeFormat(i.timestamp)
    DamageDealt+=i['damage']*((100 -numPlayers ) / 100 + 1)
    Damage_events.append((time,DamageDealt))
print("damage",Damage_events)


#击倒
DBNO_events=[]
def DBNOInTeam(o):
        i=o['attacker']['account_id']
        if i == player:
            return  i


logDBNO=list(filter(DBNOInTeam,DBNO))

DBNOs=0

for i in logDBNO:
    time=timeFormat(i.timestamp)
    DBNOs+=1
    DBNO_events.append((time,DBNOs))
print("dbno",DBNO_events)



#开始、结束时间、比赛时长
start=timeFormat(telemetry.filter_by("log_match_start")[0].timestamp)
duration=1249.859
end=start+duration

# print(start,end)

velocity_walk=float(walkDistance/duration)
velocity_drive=float(rideDistance/duration)
velocity_swim=float(swimDistance/duration)

#获取输入数据

assist_real,boost_real,damage_real,dbno_real,headshotKills_real ,heal_real,kill_real,killStreaks_real,longestKill_real ,revive_real,weapon_real=0,0,0,0,0,0,0,0,0,0,0
def get_event(event,event_real):
    j=0
    while(event!=[] and j<len(event)):

       if(event[j][0]-time<10 and event[j][0]-time>=0):
           event_real=event[j][1]
           j+=1 
       elif (event[j][0]-time>=10):
           break
       else: j+=1
    return event_real 

def get_kill_event(event,kill_real,headshotKills_real,killStreaks_real,longestKill_real):          
    j=0
    while(event!=[] and j<len(event)):
       if(event[j][0]-time<10 and event[j][0]-time>=0):
           kill_real=event[j][1]
           headshotKills_real=event[j][2]
           longestKill_real=event[j][3]
           killStreaks_real=event[j][4] 
           j+=1     
       elif (event[j][0]-time>=10):
           break
       else: j+=1
    return kill_real,headshotKills_real,longestKill_real,killStreaks_real

    #初始化数据集
real = pd.DataFrame(columns=col)
real.loc[1] =[ 0 for n in range(16)]


#合成输入数据
#代入模型
#保存结果
time=start
result=[]
while time<end:
    time+=10
    time_cost=time-start
    assist_real=get_event(assist_events,assist_real)
    boost_real=get_event(boost_events,boost_real)
    weapon_real=get_event(weapon_events,weapon_real)
    heal_real=get_event(heal_events,weapon_real)
    revive_real=get_event(revive_events,revive_real)
    dbno_real=get_event(DBNO_events,dbno_real)
    damage_real=get_event(Damage_events,damage_real)
    # print(damage_real)
    rideDistance=velocity_drive*time_cost
    swimDistance=velocity_swim*time_cost
    walkDistance=velocity_walk*time_cost
    distance=swimDistance+walkDistance+rideDistance
    kill_real,headshotKills_real,killStreaks_real,longestKill_real=get_kill_event(kill_events,kill_real,headshotKills_real,killStreaks_real,longestKill_real)
    # print(kill_real,headshotKills_real,longestKill_real,killStreaks_real)
    real.loc[1] =[assist_real,boost_real,damage_real,dbno_real,headshotKills_real ,heal_real,killPlace,kill_real,killStreaks_real,longestKill_real ,revive_real,rideDistance,swimDistance,walkDistance,weapon_real,distance]
    print(real.loc[1])
    # win_place=1-team.stats['rank']/telemetry.num_teams()
    
    win_place=1
    real_time=m1.predict(real)
    # a2=m1.score(real,[win_place])
    b2=mean_absolute_error(y_true=[win_place], y_pred=[real_time])
    e1=mean_squared_error(y_true=[win_place], y_pred=[real_time])

    print('时间',time_cost,'实际排名',win_place,"预测排名",real_time,
      '\nmean_absolute_error: ', b2,"rmae:",e1)
    result.append([time,win_place,real_time,b2,e1])      






#输出结果
name=['时间',"最终排名",'实时预测排名','平均绝对误差',"均方误差"]
test=pd.DataFrame(columns=name,data=result)
print(test)
test.to_csv('result4.csv',encoding='gbk')






