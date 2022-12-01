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
def answer_two(df):


    #df.iloc[2, 10: ].name
    avgGPD = pd.Series([])
    for i in range(len(df)):
        country_name = df.iloc[i].name
        values = df.iloc[i, 10:]
        avg = values.mean()
        avgGPD = avgGPD.append(pd.Series({ country_name : avg }))

    return avgGPD

#print(answer_two(df))



