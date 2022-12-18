import pandas as pd
import numpy as np
import Dataset
import DataAnalys
import matplotlib.pyplot as plt

def FillAge(df):

    df['Age'].fillna(df['Age'].mean(), inplace=True)

def changeSexToNumericValues(df):

    df['Sex'] = df['Sex'].replace(['male'], 0)
    df['Sex'] = df['Sex'].replace(['female'], 1)

def FillEmbarked(df):

    df['Embarked'].fillna(0, inplace = True);
def changeEmbarkedToNumericValues(df):

    df['Embarked'] = df['Embarked'].replace(['S'], 0)
    df['Embarked'] = df['Embarked'].replace(['C'], 1)
    df['Embarked'] = df['Embarked'].replace(['Q'], 2)


def getFamilyLen(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']

    return df

def formatNames(df):
        df.Name = df.Name.str.extract('\, ([A-Z][^ ]*\.)', expand=False)

        df["Name"].fillna('Mr', inplace=True)

        return df;

def PinAge(df):

    df = df.mask(df['Age'] <= 5, 0)
    #df = df.mask((df['Age'] >= 16) & (df["Age"]<= 30), 70)



    return df;
def changeNamesToNumericValues(df):

    names = df["Name"].value_counts()
    names = {v: k for k, v in enumerate(names.keys())}
    df["Name"] = df["Name"].apply(lambda x: names[x]);
    return df;
def FillFare(df):
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
def preprocessing(df):

    features = Dataset.Dataset().GetFeatures();
    # Expecting only 10 features without "Survived", PassengerId should be index;
    df = df[features]
    FillAge(df)
    FillEmbarked(df)
    FillFare(df)
    changeSexToNumericValues(df)
    changeEmbarkedToNumericValues(df)
    df = getFamilyLen(df)
    df = formatNames(df)
    df = changeNamesToNumericValues(df)
    df = PinAge(df)
    return df;


#titanic = Dataset.Dataset()

#x = titanic.getTrainData()
#x = preprocessing(x)
#PinAge(x)
#DataAnalys.VisualizeDependency(x, 'Name', "Survived")
#plt.show()