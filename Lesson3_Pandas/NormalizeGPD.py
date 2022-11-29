import pandas as pd
import numpy as np
import re
import NormalizeEnergyDataset
import csv






class DataFrame:

    def get_df(self):

        pass;

class GPDDataFrame(DataFrame):

    def get_df(self):

        return pd.read_csv("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4701247.csv", sep=",", skiprows=4)

class Decorator(DataFrame):

    def __init__(self, GPDDataFrame):

        self._df = GPDDataFrame;

    def get_df(self):

        return self._df.get_df();

class RenameCountryDecorator(Decorator):
    def RenameCountry(self, df):
        CountryDict = {
            "Korea, Rep.": "South Korea",
            "Iran, Islamic Rep.": "Iran",
            "Hong Kong SAR, China": "Hong Kong"
        }
        index = 0
        for name in df["Country name"]:
            if name in CountryDict.keys():
                df.loc[index, "Country name"] = CountryDict[name]
            index += 1;
        return df

    def get_df(self):

        df = self._df.get_df()
        return self.RenameCountry(df)

class RemoveUselessDataDecorator(Decorator):

    def RemoveUselesDataFromDataFrame(self, dff):

        for i in range(1960, 2023):
            year = i

            if (year < 2006 or year > 2015):
                del dff[str(year)]

        del dff["Country Code"]
        del dff["Indicator Name"]
        del dff["Indicator Code"]

        return dff

    def get_df(self):

        df=self._df.get_df()

        return self.RemoveUselesDataFromDataFrame(df);

def GetGPD():
    gpd = GPDDataFrame()
    decorator1 = RenameCountryDecorator(gpd)
    decorator2 = RemoveUselessDataDecorator(decorator1)

    return decorator2.get_df()

