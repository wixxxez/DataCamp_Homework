import pandas as pd
import numpy as np
import re
import MergeService
import csv
import HelpService

def answer_one():

    dataFrame = MergeService.getMergedFrames()

    return dataFrame

df = answer_one()

#print(df)
def answer_two():

    df = answer_one()
    #df.iloc[2, 10: ].name
    avgGPD = pd.Series([])
    for i in range(len(df)):
        country_name = df.iloc[i].name
        values = df.iloc[i, 10:]
        avg = values.mean()
        avgGPD = avgGPD.append(pd.Series({ country_name : avg }))

    return avgGPD.sort_values(ascending=False)

#print(answer_two())

def answer_three():
    top15 = answer_one()
    top15Avg = answer_two();

    countryAvg = top15Avg.iloc[5]
    countryName = top15Avg[top15Avg == countryAvg].index[0]

    return top15.loc[countryName]["2015"] - top15.loc[countryName]["2006"]

#print(answer_three())

def answer_four():
    top15 = answer_one()
    coef = []
    for i in range(len(top15)):
        coef.append(top15.iloc[i]["Citations"] / top15.iloc[i]["Self-citations"])
    top15['coef'] = coef
    top15 = top15.sort_values(by="coef", ascending=False)

    return (top15.iloc[0].name, top15.iloc[0]["coef"])

#print(answer_four())

def answer_five():

    top15 = answer_one()

    populations = [];
    for i in range(len(top15)):
        EnergySupply = top15.iloc[i]["Energy Supply"]
        EnergySupplyPerCapita = top15.iloc[i]["Energy Supply per capita"];
        populations.append(EnergySupply/EnergySupplyPerCapita)
    top15["Populations"] = populations
    top15 = top15.sort_values(by="Populations", ascending=False)
    return top15.iloc[2].name
print(answer_five())