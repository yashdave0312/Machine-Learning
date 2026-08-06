import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("insurance.csv")
print(df)

print(df.shape)
print(df.info())
print(df.describe())

print(df.isnull().sum())

numeric =["age", "bmi", "charges"]
for col in numeric:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col],kde=True,bins = 20)
    plt.show()

categorical = ["sex", "children", "smoker", "region"]
for col in categorical:
    plt.figure(figsize=(6,4))
    sns.countplot(x = df[col])
    plt.show()