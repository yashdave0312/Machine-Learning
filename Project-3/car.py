import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr,chi2_contingency

warnings.filterwarnings("ignore")

df = pd.read_csv("Car_prediction.csv")

# EDA

# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

# print(df.drop_duplicates())

# print(df.isnull().sum())
# print(df.dtypes)

numerical = ["year","price","mileage","tax","mpg","engineSize"]

# for cols in numerical :
#     sns.histplot(x = df[cols],kde = True)
#     plt.show()

print(df.loc[df["engineSize"] == 0,"engineSize"].count())
print(df.columns)

categorical = ["model","transmission","fuelType"]
# for cols in categorical:
#     sns.countplot(x = df[cols])
#     plt.show()

print(df.loc[df["fuelType"] == "Electric","fuelType"].count())

print(df["model"].value_counts())

# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

# Data Preprocessing started

print(categorical)
print(df["fuelType"].value_counts())
print(df["transmission"].value_counts())

df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True)
print(df_encoded.head())

df_encoded = df_encoded.astype(int)
print(df_encoded.head())

# Scalarisation

scaler = StandardScaler()
scal_cols =["year","mileage","tax","mpg","engineSize"]

df_encoded[scal_cols] = scaler.fit_transform(df_encoded[scal_cols])
print(df_encoded.head())


# Pearson Correlation