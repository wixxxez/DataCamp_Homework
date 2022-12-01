import pandas as pd
import  numpy as np

import  NormalizeGPD
import EnergyDataFrameBuilder

def Merge(df1,df2,df3,key):


    df4 = pd.merge(df1,df2,on = key,)
    df5 = pd.merge(df3,df4, on = key)
    del df5["Region"]
    return df5
def getMergedFrames():

    energy = EnergyDataFrameBuilder.GetEnergyDataFrame()
    GPD = NormalizeGPD.GetGPD()
    ScimEn = pd.read_excel("scimagojr country rank 1996-2021.xls");
    df =Merge(energy,GPD,ScimEn, "Country name")

    return df.set_index("Country name").head(15)

