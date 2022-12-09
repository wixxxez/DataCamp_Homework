import pandas as pd

def get_top15withPopulations(top15):
    populations = [];
    for i in range(len(top15)):
        EnergySupply = top15.iloc[i]["Energy Supply"]
        EnergySupplyPerCapita = top15.iloc[i]["Energy Supply per capita"];

        populations.append(EnergySupply / EnergySupplyPerCapita)
    top15["Populations"] = populations

    return top15;

def GetContinents(ContinentDict, main_df):
    Continent = []
    frame = []
    for i in range(len(main_df)):
        name = main_df.iloc[i].name
        frame.append(ContinentDict[name])
        frame.append(name)
        frame.append(main_df.iloc[i]["Populations"])
        Continent.append(frame)
        frame = []

    return Continent

def getSize(df):
    size = df.groupby("Continent").count()
    del size["Populations"]
    return size;

def getStd(df):
    std = [
        ['Asia', df.query("Continent == 'Asia'").std()[0]],
        ['Europe', df.query("Continent == 'Europe'").std()[0]],
        ['Australia', df.query("Continent == 'Australia'").std()[0]],
        ['North America', df.query("Continent == 'North America'").std()[0]],
        ['South America', df.query("Continent == 'South America'").std()[0]]
    ]
    std = pd.DataFrame(std, columns=["Continent", "std"])
    return std;
class Merge:

    def merge(self,df, df2, newname):
        pass;

class MultipleMerge(Merge):

    def merge(self,df, df2, newname):

        dataframe = pd.merge(df,df2, on = "Continent")
        dataframe.rename(columns={"Populations": newname}, inplace=True)
        return dataframe



