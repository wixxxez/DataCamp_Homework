import pandas as pd
import numpy as np
import re
import NormalizeEnergyDataset
import csv


def RenameCountry(df):
    CountryDict = {
        "Korea, Rep.": "South Korea",
        "Iran, Islamic Rep.": "Iran",
        "Hong Kong SAR, China": "Hong Kong"
    }
    index = 0
    for name in df["Country name"]:
        if name in CountryDict.keys():
            df.loc[index,"Country name"] = CountryDict[name]
        index +=1;
    return df

def RemoveUselesDataFromDataFrame(dff):

    for i in range(1960,2023):
        year = i

        if(year < 2006 or  year > 2015):

            del dff[str(year)]

    del dff["Country Code"]
    del dff["Indicator Name"]
    del dff["Indicator Code"]

    return dff

def GetGPD():
    gpd = pd.read_csv("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4701247.csv", sep=",", skiprows=4)
    gpd = RenameCountry(gpd)
    gpd = RemoveUselesDataFromDataFrame(gpd)
    return gpd
