import pandas as pd
import numpy as np
import re
#load data









class DataFrame:

    def get_df(self):

        pass

class EnergyDataFrame(DataFrame):

    def get_df(self):

        return pd.read_excel('Energy Indicators.xls', sheet_name="Energy", skiprows=16, usecols="C:F", nrows=228)

class Decorator(DataFrame):

    def __init__(self, DataFrame: DataFrame) -> None:

        self._df = DataFrame
    def get_df(self):

        return self._df.get_df();

class NormalizeEnergySupplyDecorator(Decorator):

    def normalizeEnergySupply(self, df):
        index = 0;
        for i in df["Energy Supply"]:
            if i == "Petajoules":
                # print("Convert Petajoules to Gigajoules procces ")
                pass
            elif i == "...":
                df.loc[index, "Energy Supply"] = np.NaN
            else:
                i = i * 1000000
                df.loc[index, "Energy Supply"] = i
            index += 1;
        return df;

    def get_df(self):

        df = self._df.get_df();
        return self.normalizeEnergySupply(df);

class normalizeEnergyPerCapitaDecorator(Decorator):

    def normalizeEnergySupplyPerCapita(self,df):
        index = 0;
        for i in df["Energy Supply per capita"]:
            if i == "Petajoules":
                print("Convert Petajoules to Gigajoules procces ")
                pass
            elif i == "...":
                df.loc[index, "Energy Supply per capita"] = np.NaN
            index += 1;
        return df;

    def get_df(self):

        df = self._df.get_df()
        return self.normalizeEnergySupplyPerCapita(df)

class RenameCountriesDecorator(Decorator):

    def RenameCountries(self,df):
        ContruiesDict = {
            "Republic of Korea": "South Korea",
            "United States of America": 'United States',
            'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
            'China, Hong Kong Special Administrative Region': 'Hong Kong'
        }

        index = 0
        for i in df['Country name']:
            if (i in ContruiesDict.keys()):
                df.loc[index, "Country name"] = ContruiesDict[i]
            index += 1

        return df

    def get_df(self):

        df = self._df.get_df()
        return self.RenameCountries(df);

class RemoveNumbersFromCountryNameDecorator(Decorator):

    def RemoveNumbersFromCountryName(self,df):
        index = 0
        for name in df['Country name']:
            if type(name) != float:
                newname = ''.join([i for i in name if not i.isdigit()])
                df.loc[index, "Country name"] = newname;
                index += 1

        return df;

    def get_df(self):
        df = self._df.get_df()
        return self.RemoveNumbersFromCountryName(df);

class RemoveTextInBracketsFromCountryName(Decorator):

    def RemoveTextFromBracketsInCountryName(self, df):
        index = 0;
        for name in df['Country name']:
            name = re.sub("\(.*?\)", "", name)
            df.loc[index, "Country name"] = name
            index += 1
        return df

    def get_df(self):

        df = self._df.get_df()
        return self.RemoveTextFromBracketsInCountryName(df);

def GetEnergy():

    energyDataFrame = EnergyDataFrame();

    print(energyDataFrame.get_df())
    #energy = pd.read_excel('Energy Indicators.xls', sheet_name="Energy", skiprows=16, usecols="C:F", nrows=228)
    #energy = normalizeEnergySupply(energy)
    #energy =RemoveNumbersFromCountryName(energy)
    #energy = normalizeEnergySupplyPerCapita(energy)
    #energy = RenameCountries(energy)

    #energy = RemoveTextFromBracketsInCountryName(energy)
    #return energy;

GetEnergy()