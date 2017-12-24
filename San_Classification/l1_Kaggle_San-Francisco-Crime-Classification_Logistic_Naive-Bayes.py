#!/usr/bin/python3
# coding: utf-8
# 参考: [NLP 系列(4)_朴素贝叶斯实战与进阶](http://blog.csdn.net/han_xiaoyang/article/details/50629608)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import time
##################################################################
## 一: 数据预览
# Dates - timestamp of the crime incident; 日期
# Category - category of the crime incident (only in train.csv). This is the target variable you are going to predict.; 犯罪类型, 比如 Larceny/盗窃罪
# Descript - detailed description of the crime incident (only in train.csv); 对于犯罪更详细的描述
# DayOfWeek - the day of the week; 星期几
# PdDistrict - name of the Police Department District; 所属警区
# Resolution - how the crime incident was resolved (only in train.csv); 处理结果, 比如说『逮捕』『逃了』
# Address - the approximate street address of the crime incident; 发生街区位置
# X - Longitude; GPS 坐标
# Y - Latitude
#用 pandas 载入 csv 训练数据, 并解析第一列为日期格式
train = pd.read_csv('./tmp_dataset/Kaggle-San-Francisco-Crime-Classification/train.csv', parse_dates=['Dates'])  # 并解析第一列为日期格式
test = pd.read_csv('./tmp_dataset/Kaggle-San-Francisco-Crime-Classification/test.csv', parse_dates=['Dates'])
print(train.columns.values)  # ['Dates' 'Category' 'Descript' 'DayOfWeek' 'PdDistrict' 'Resolution' 'Address' 'X' 'Y']
print(train.head()) # 读取前几行
print(train.info())
# Dates         878049 non-null datetime64[ns]
# Category      878049 non-null object
# Descript      878049 non-null object
# DayOfWeek     878049 non-null object
# PdDistrict    878049 non-null object
# Resolution    878049 non-null object
# Address       878049 non-null object
# X             878049 non-null float64
# Y             878049 non-null float64

##################################################################
## 二: 数据预处理
# 用 LabelEncoder **对犯罪类型做编号**
# 处理时间, 在我看来, 也许犯罪发生的时间点(小时)是非常重要的, 因此我们会用 Pandas 把这部分数据抽出来
# 对街区, 星期几, 时间点用 get_dummies()one-hot因子化
# 做一些组合特征, 比如把上述三个 feature 拼在一起, 再因子化一下

# 用 LabelEncoder 对不同的犯罪类型编号
leCrime = preprocessing.LabelEncoder()
crime = leCrime.fit_transform(train.Category);
print(train.Category[:2])  # 0          WARRANTS; 1    OTHER OFFENSES
print(len(crime), len(set(crime)))  # 878049 39 set去重查看种类
# 因子化星期几, 街区, 小时等特征
days = pd.get_dummies(train.DayOfWeek); print(days.columns.values)  # ['Friday' 'Monday' 'Saturday' 'Sunday' 'Thursday' 'Tuesday' 'Wednesday']
district = pd.get_dummies(train.PdDistrict); print(district.columns.values)  # ['BAYVIEW' 'CENTRAL' 'INGLESIDE' 'MISSION' 'NORTHERN' 'PARK' 'RICHMOND' 'SOUTHERN' 'TARAVAL' 'TENDERLOIN']
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour); print(hour.columns.values)  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
# 组合特征
trainData = pd.concat([hour, days, district], axis=1)
trainData['crime'] = crime;
print(trainData.head())
print(len(trainData.columns.values))  # 42; 竟然分出了这么多的特征...
# 对于测试数据做同样的处理
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)
testData = pd.concat([hour, days, district], axis=1)
print(len(trainData.columns.values))  # 42

##################################################################
## 三: 建立模型
# 还需要提到的一点是, 大家参加 Kaggle 的比赛, 一定要注意最后排名和评定好坏用的标准, 比如说在现在这个多分类问题中,
#     Kaggle 的评定标准并不是 precision, 而是 multi-class log_loss, 这个值越小, 表示最后的效果越好.
# 我们可以快速地筛出一部分重要的特征, 搭建一个 baseline 系统, 再考虑步步优化. 比如我们这里简单一点, 就只取星期几和街区作为分类器输入特征,
#     我们用 scikit-learn 中的 train_test_split 函数拿到训练集和交叉验证集, 用朴素贝叶斯和逻辑回归都建立模型, 对比一下它们的表现:
features = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
            'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']  # 只取星期几和街区作为分类器输入特征
# 下面这行是添加 hour 属性, 只要取消注释即可
# features = features + [x for x in range(0, 24)]  # 添加犯罪的小时时间点作为特征
training, validation = train_test_split(trainData, test_size=.40)  # 分割训练集(3/5)和测试集(2/5)
##################################################################
## 朴素贝叶斯建模, 计算 log_loss
model = BernoulliNB()
nbStart = time.time()
model.fit(training[features], training['crime'])  # 这个很快
nbCostTime = time.time() - nbStart
predicted = np.array(model.predict_proba(validation[features]))
print("朴素贝叶斯建模耗时 %f 秒" % (nbCostTime))
print("朴素贝叶斯 log 损失为 %f" % (log_loss(validation['crime'], predicted)))  # 2.617892 / 2.582167; 后者是将上面的 特征加入 hour 后
##################################################################
## 逻辑回归建模, 计算 log_loss
model = LogisticRegression(C=.01)
lrStart= time.time()
model.fit(training[features], training['crime'])  # 这个就很慢了
lrCostTime = time.time() - lrStart
predicted = np.array(model.predict_proba(validation[features]))
print("逻辑回归建模耗时 %f 秒" % (lrCostTime))  # 近 2min
print("逻辑回归 log 损失为 %f" % (log_loss(validation['crime'], predicted)))  # 2.624773 / 2.592119; 还没 NB 好, 后者是将上面的 特征加入 hour 后

# 可以看到在这三个类别特征下, 朴素贝叶斯相对于逻辑回归, 依旧有一定的优势(log 损失更小), 同时训练时间很短, 这意味着模型虽然简单, 但是效果依旧强大
# 顺便提一下, 朴素贝叶斯 1.13s 训练出来的模型, 预测的效果在 Kaggle 排行榜上已经能进入 Top 35%了, 如果进行一些优化,
#     比如特征处理、特征组合等, 结果会进一步提高
#用 LabelEncoder 对不同的犯罪类型编号

##################################################################
## Kaggle提交测试集
# 以前10为例
model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted_test=pd.DataFrame(model.predict_proba(testData[features][:10]), columns=list(leCrime.inverse_transform(model.classes_))) # 不加list转换也行
# type(leCrime.inverse_transform(model.classes_))为np.ndarray,若['a']+array['b','c']->['ab','ac'];['a']+list['b','c']->['a','b','c']
# predicted_test=pd.DataFrame(model.predict_proba(testData[features]), columns=list(leCrime.inverse_transform(model.classes_))) # 不加list转换也行
predicted_test.index.name = "Id"  # 满足输出格式
predicted_test.to_csv("./tmp_dataset/Kaggle-San-Francisco-Crime-Classification/out.csv", float_format='%.1f')

# 或者
model = BernoulliNB()
model.fit(train_data[features], train_data['crime'])
predicted = model.predict_proba(test_data[features])
#Write results
result=pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv('testResult.csv', index = True, index_label = 'Id' )

##################################################################
## 改进
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
features2 = [x for x in range(0,24)]
features = features + features2  # [hour, days, district]
