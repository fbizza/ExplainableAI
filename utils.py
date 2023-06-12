import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

def import_boston_dataset():
    boston = fetch_openml(name='boston')
    X = boston.data
    y = boston.target
    features_names = boston.feature_names
    return X, y, features_names

def import_compas_dataset():
    df = pd.read_csv('../datasets/propublica_data_for_fairml.csv')
    X = df.drop(columns=['Two_yr_Recidivism'])
    y = df['Two_yr_Recidivism']
    features_names = list(X.columns.values)
    return X, y, features_names

def import_income_dataset():
    df = pd.read_csv("../datasets/income.csv")
    df = df.drop(df.columns[[0, 2, 4, 10, 11, 12, 13]], axis=1) # Not useful features
    label_encoder = LabelEncoder()
    df['class'] = df['class'].map({'<=50K': 0, '>50K': 1}).astype(int)
    df['marital-status'] = df['marital-status'].map(
        {'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3,
         'Divorced': 4, 'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
    df.workclass = label_encoder.fit_transform(df.workclass)
    df.race = label_encoder.fit_transform(df.race)
    df.sex = label_encoder.fit_transform(df.sex)
    df.occupation = label_encoder.fit_transform(df.occupation)
    df.education = label_encoder.fit_transform(df.education)
    df.relationship = label_encoder.fit_transform(df.relationship)
    X = df
    y = X.iloc[:, 7]
    X = X.iloc[:, :-1]
    features_names = list(X.columns.values)
    return X, y, features_names
