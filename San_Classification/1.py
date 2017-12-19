#!/usr/bin/python3
# coding: utf-8
import pandas as pd
import numpy as np

#用pandas载入csv训练数据，并解析第一列为日期格式
train=pd.read_csv('./tmp_data/San_Francisco_Crime_Classification/train.csv', parse_dates = ['Dates'])
test=pd.read_csv('./tmp_data/San_Francisco_Crime_Classification/test.csv', parse_dates = ['Dates'])
print(train)
