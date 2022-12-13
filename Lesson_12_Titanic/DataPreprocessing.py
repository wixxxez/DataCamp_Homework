import pandas as pd
import numpy as np
import Dataset
import DataAnalys


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

def preprocessing(df):

    features = Dataset.Dataset().GetFeatures();
    # Expecting only 10 features without "Survived", PassengerId should be index;
    df = df[features]
    FillAge(df)
    FillEmbarked(df)
    changeSexToNumericValues(df)
    changeEmbarkedToNumericValues(df)

    return df;


titanic = Dataset.Dataset()

X_train = titanic.getXTrain()
Y_train = titanic.getYTrain()
X_train= preprocessing(X_train)
print(X_train.info())
print(DataAnalys.checkNAN(X_train))