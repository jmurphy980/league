# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:15:39 2022

@author: jmurphy
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("high_diamond_ranked_10min.csv")
# Lists the top of the data frame
print(df.head())
# Lists the columns in the data frame
print(df.columns)

# Define the target variable
y = df.blueWins

# Define the variables we'll be using to predict the target
feature_names = ['blueWardsPlaced', 'blueWardsDestroyed',
       'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
       'blueEliteMonsters', 'blueDragons', 'blueHeralds',
       'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
       'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff',
       'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed',
       'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
       'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
       'redTotalGold', 'redAvgLevel', 'redTotalExperience',
       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff',
       'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin']

x = df[feature_names]

league_model = DecisionTreeRegressor(random_state=1)

league_model.fit(x,y)

prediction = league_model.predict(x)

print(prediction)