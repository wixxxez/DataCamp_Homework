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
print("---Answer One---")
print(df)
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

print("---Answer Two---")
print(answer_two())

def answer_three():
    top15 = answer_one()
    top15Avg = answer_two();

    countryAvg = top15Avg.iloc[5]
    countryName = top15Avg[top15Avg == countryAvg].index[0]

    return top15.loc[countryName]["2015"] - top15.loc[countryName]["2006"]

print("---Answer Three---")
print(answer_three())

def answer_four():
    top15 = answer_one()
    coef = []
    for i in range(len(top15)):
        coef.append(top15.iloc[i]["Self-citations"] / top15.iloc[i]["Citations"])
    top15['coef'] = coef
    top15 = top15.sort_values(by="coef", ascending=False)

    return (top15.iloc[0].name, top15.iloc[0]["coef"])

print("---Answer Four---")
print(answer_four())

def answer_five():

    top15 = answer_one()

    top15 = HelpService.get_top15withPopulations(top15)

    top215 = top15.sort_values(by="Populations", ascending=False)

    return top215.iloc[2].name

print("---Answer Five---")
print(answer_five())

def answer_six():

    top15 = answer_one();
    top15 = HelpService.get_top15withPopulations(top15)

    Populations = top15.Populations

    corr_df = {
        "Populations": Populations,
        "Citable documents per capita": top15["Citable documents"]
    }

    return pd.DataFrame(corr_df).corr(method='pearson').Populations[1]

def answer_seven():

    top15 = HelpService.get_top15withPopulations(top15=answer_one());
    ContinentDict = {'China': 'Asia',
                     'United States': 'North America',
                     'Japan': 'Asia',
                     'United Kingdom': 'Europe',
                     'Russian Federation': 'Europe',
                     'Canada': 'North America',
                     'Germany': 'Europe',
                     'India': 'Asia',
                     'France': 'Europe',
                     'South Korea': 'Asia',
                     'Italy': 'Europe',
                     'Spain': 'Europe',
                     'Iran': 'Asia',
                     'Australia': 'Australia',
                     'Brazil': 'South America'}

    Continent = HelpService.GetContinents(ContinentDict,top15);

    df = pd.DataFrame(Continent, columns=["Continent","size", "Populations"])
    size =HelpService.getSize(df)

    sum = df.groupby("Continent").sum("Populations")
    mean = df.groupby("Continent").mean("Populations")
    std = HelpService.getStd(df);

    MergeService = HelpService.MultipleMerge()
    df = MergeService.merge(size,sum, "sum");
    df = MergeService.merge(df,mean,"mean")
    df = MergeService.merge(df,std,"std")
    return df.set_index("Continent")

print("---Answer Six---")
print(answer_six());

print("---Answer Seven---")
print(answer_seven())