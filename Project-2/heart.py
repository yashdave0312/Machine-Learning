import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

df = pd.read_csv("heart.csv")
print(df.head())

print(df.shape)
print(df.describe())
print(df.columns)
print(df.describe())

#EDA 

print(df.duplicated().sum())
print(df["HeartDisease"].value_counts())

print(df.isnull().sum())

def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True,bins=20)
    plt.show()

plotting("Age",1)
plotting("RestingBP",2)
plotting("Cholesterol",3)
plotting("MaxHR",4)
plt.tight_layout()

ch_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
df["Cholesterol"] = df["Cholesterol"].replace(0,ch_mean)
print(df["Cholesterol"].head())

restbp_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()
df["RestingBP"] = df["RestingBP"].replace(0,restbp_mean)

categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

for col in categorical:
    plt.figure(figsize=(6,4))
    sns.countplot(x = df[col], hue = df["HeartDisease"])
    plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# Data  Preprocessing
df_encoded = pd.get_dummies(df, columns=categorical, drop_first=True)
print(df_encoded.head())

df_encoded = df_encoded.astype(int)
print(df_encoded.head())

# scalarisation
scalar = StandardScaler()
scale_cols = ["Age","RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

df_encoded[scale_cols] = scalar.fit_transform(df_encoded[scale_cols])
print(df_encoded.head())
